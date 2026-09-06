"""
stt.py — Speech-to-Text using faster-whisper, for both batch files and
real-time audio streams.

Transcription never forces a language parameter so English/Hindi/Hinglish
code-switched speech is handled naturally.

Two model tiers serve the realtime path:

    PARTIAL tier ("tiny" by default)
        Runs repeatedly on the *same growing utterance* while the clinician
        is still speaking, to paint interim text on screen. Accuracy matters
        less than turnaround, because every partial is thrown away and
        replaced by the next one.

    FINAL tier ("base" on CPU, "large-v3" on CUDA)
        Runs once per utterance after the VAD closes it. This text is what
        gets stored, diarized and fed to the clinical extractor, so it uses
        the largest model the hardware can afford.

Both tiers share one cache, so a deployment that sets both to the same size
pays for a single model in memory.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch
from faster_whisper import WhisperModel

# ── Model cache ──────────────────────────────────────────────────────────
_models: dict[str, WhisperModel] = {}
_model_lock = threading.Lock()

# Whisper's own encoder is not safe to call concurrently from several
# threads, so every transcription serialises on this lock. The realtime
# engine is built around that constraint: it drops stale partials rather
# than queueing them up behind the lock.
_inference_lock = threading.Lock()


def default_final_model() -> str:
    """Model size used for committed transcript text."""
    override = os.getenv("WHISPER_FINAL_MODEL")
    if override:
        return override.strip()
    return "large-v3" if torch.cuda.is_available() else "base"


def default_partial_model() -> str:
    """Model size used for throwaway interim text during speech."""
    override = os.getenv("WHISPER_PARTIAL_MODEL")
    if override:
        return override.strip()
    return "small" if torch.cuda.is_available() else "tiny"


def _get_model(model_size: str | None = None) -> WhisperModel:
    """
    Lazily load a faster-whisper model and cache it by size.

    Defaults to `default_final_model()`. CUDA runs float16; CPU runs int8,
    which keeps the base model fast enough for interactive use.
    """
    if model_size is None:
        model_size = default_final_model()

    cached = _models.get(model_size)
    if cached is not None:
        return cached

    with _model_lock:
        # Re-check inside the lock: two streams starting at once must not
        # each pay to load the same weights.
        cached = _models.get(model_size)
        if cached is not None:
            return cached

        if torch.cuda.is_available():
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
        else:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")

        _models[model_size] = model
        return model


def preload_models(sizes: list[str] | None = None) -> None:
    """
    Warm the model cache ahead of the first request.

    Called at application startup so the first utterance of the first
    consultation is not delayed by several seconds of weight loading.
    """
    targets = sizes or [default_partial_model(), default_final_model()]
    for size in dict.fromkeys(targets):
        try:
            _get_model(size)
        except Exception as e:  # pragma: no cover - depends on local weights
            print(f"[stt] Could not preload Whisper '{size}': {e}")


def _segments_to_dicts(segments_iter, time_offset: float = 0.0) -> list[dict]:
    """Materialise a faster-whisper segment generator into plain dicts."""
    out = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg.start) + time_offset, 3),
            "end": round(float(seg.end) + time_offset, 3),
            "text": text,
        })
    return out


# ══════════════════════════════════════════════════════════════════════════
# BATCH (FILE) TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════════════

def transcribe(audio_path: str, model_size: str | None = None) -> list[dict]:
    """
    Transcribe an audio file into timestamped text segments.

    Args:
        audio_path: Path to the audio file (wav, mp3, etc.).
        model_size: Optional whisper model size ('tiny', 'base', 'small', 'medium').

    Returns:
        List of dicts with keys: start (float, seconds), end (float, seconds),
        text (str). No forced language — auto-detection supports code-switching.
    """
    model = _get_model(model_size)

    with _inference_lock:
        # language=None → auto-detect; supports code-switched En/Hi/Hinglish
        segments_iter, _info = model.transcribe(
            audio_path,
            vad_filter=True,
            word_timestamps=False,
            language=None,
        )
        return _segments_to_dicts(segments_iter)


# ══════════════════════════════════════════════════════════════════════════
# STREAMING (IN-MEMORY) TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════════════

def transcribe_array(
    samples: np.ndarray,
    model_size: str | None = None,
    time_offset: float = 0.0,
    partial: bool = False,
    initial_prompt: str | None = None,
) -> list[dict]:
    """
    Transcribe raw 16 kHz mono float32 samples already held in memory.

    This is the realtime entry point: audio arrives from the WebSocket as
    PCM and is handed straight to Whisper with no file ever created.

    Args:
        samples:        Mono float32 audio in [-1, 1] at 16 kHz.
        model_size:     Explicit model size; defaults to the partial or final
                        tier depending on `partial`.
        time_offset:    Seconds to add to every timestamp so segment times
                        are absolute within the consultation rather than
                        relative to this buffer.
        partial:        True for interim in-progress decodes. Enables greedy,
                        single-beam decoding and skips Whisper's internal VAD,
                        which together roughly halve turnaround.
        initial_prompt: Preceding transcript text used as decoder context.
                        This is what keeps terminology and spelling stable
                        across utterance boundaries.

    Returns:
        List of {start, end, text} dicts; empty when nothing was recognised.
    """
    if samples is None or samples.size == 0:
        return []

    audio = np.ascontiguousarray(samples, dtype=np.float32)

    if model_size is None:
        model_size = default_partial_model() if partial else default_final_model()
    model = _get_model(model_size)

    options: dict = {
        "language": None,
        "word_timestamps": False,
        "initial_prompt": initial_prompt or None,
        "condition_on_previous_text": False,
    }
    if partial:
        options.update(
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
        )
    else:
        options.update(
            beam_size=5,
            vad_filter=True,
        )

    with _inference_lock:
        segments_iter, _info = model.transcribe(audio, **options)
        return _segments_to_dicts(segments_iter, time_offset=time_offset)


def transcribe_text(
    samples: np.ndarray,
    model_size: str | None = None,
    partial: bool = False,
    initial_prompt: str | None = None,
) -> str:
    """Convenience wrapper returning only the joined transcript text."""
    segments = transcribe_array(
        samples,
        model_size=model_size,
        partial=partial,
        initial_prompt=initial_prompt,
    )
    return " ".join(seg["text"] for seg in segments).strip()
