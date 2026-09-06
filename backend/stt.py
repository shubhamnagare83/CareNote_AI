"""
stt.py — Speech-to-Text using faster-whisper.

Transcribes audio without forcing a language parameter so that
English/Hindi/Hinglish code-switched speech is handled naturally.
Falls back to CPU + int8 + medium model when no GPU is available.
"""

import torch
from faster_whisper import WhisperModel

# ── Model cache ──────────────────────────────────────────────────────────
_models: dict[str, WhisperModel] = {}


def _get_model(model_size: str | None = None) -> WhisperModel:
    """
    Lazily load the faster-whisper model and cache it.
    Default on CPU is 'base' for lightning fast 2-4s turnaround (medium was 60s+).
    On GPU with CUDA, defaults to 'large-v3'.
    """
    global _models
    if model_size is None:
        model_size = "large-v3" if torch.cuda.is_available() else "base"

    if model_size in _models:
        return _models[model_size]

    if torch.cuda.is_available():
        _models[model_size] = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16",
        )
    else:
        _models[model_size] = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

    return _models[model_size]


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

    # language=None → auto-detect; supports code-switched En/Hi/Hinglish
    segments_iter, info = model.transcribe(
        audio_path,
        vad_filter=True,
        word_timestamps=False,
        language=None,
    )

    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })

    return segments
