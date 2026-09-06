"""
diarize.py — Speaker diarization via pyannote.audio + merge with transcript.

Uses the pyannote/speaker-diarization-3.1 pretrained pipeline (requires
HF_TOKEN env var with accepted model terms on HuggingFace).
"""

import os
from pyannote.audio import Pipeline

# ── Pipeline singleton ───────────────────────────────────────────────────
_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    """Lazily load the pyannote speaker-diarization pipeline once."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[diarize] HF_TOKEN not set; skipping pyannote neural diarization for speed.")
        return None

    try:
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as e:
        print(f"[diarize] PyAnnote pipeline load error: {e}. Using fast fallback.")
        _pipeline = None

    return _pipeline


def diarize(audio_path: str) -> list[dict]:
    """
    Run speaker diarization on an audio file.
    If HF_TOKEN is absent or pyannote is unavailable, returns an empty list
    so the pipeline immediately relies on fast LLM semantic speaker assignment.
    """
    pipeline = _get_pipeline()
    if pipeline is None:
        return []

    try:
        diarization = pipeline(audio_path)
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            })
        return turns
    except Exception as e:
        print(f"[diarize] Diarization runtime error: {e}. Falling back to fast mode.")
        return []


def merge_transcript_and_diarization(
    segments: list[dict],
    turns: list[dict],
) -> list[dict]:
    """
    Assign a speaker label to each transcript segment based on maximum
    time overlap with diarization turns.

    Args:
        segments: Transcript segments from stt.transcribe().
                  Each has keys: start, end, text.
        turns:    Diarization turns from diarize().
                  Each has keys: start, end, speaker.

    Returns:
        Merged list of dicts with keys: speaker, start, end, text.
        Speaker defaults to "UNKNOWN" if no diarization turn overlaps.
    """
    merged = []

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for turn in turns:
            # Calculate overlap between transcript segment and diarization turn
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        merged.append({
            "speaker": best_speaker,
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"],
        })

    return merged
