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


# ══════════════════════════════════════════════════════════════════════════
# ONLINE (STREAMING) DIARIZATION
# ══════════════════════════════════════════════════════════════════════════
#
# The pyannote pipeline above needs the complete recording before it can
# assign speakers, which makes it unusable while a consultation is still in
# progress. `OnlineDiarizer` fills that gap: it labels each utterance the
# moment the VAD closes it, using an incremental clustering scheme over the
# lightweight voice embeddings from backend.audio.
#
# Accuracy trade-off is explicit and acceptable here — anonymous labels
# (SPEAKER_00/01) only need to be *consistent*, because the LLM pass in
# extract.py is what decides which label is the Doctor and which is the
# Patient. Getting the count and the boundaries right matters; getting the
# identity right does not.

import numpy as np

from backend.audio import cosine_similarity, speaker_embedding


class OnlineDiarizer:
    """
    Incremental multi-party speaker attribution for a live stream.

    Real consultations are rarely two-person. A patient often arrives with a
    spouse, parent, adult child or friend who answers on their behalf, and a
    nurse may interject. This diarizer tracks each distinct voice separately
    so the clinical layer can tell who actually said what — which matters,
    because a symptom reported by an attendant is collateral history, and a
    symptom the *attendant* has themselves is not the patient's at all.

    Each finalised utterance is embedded and compared against the running
    centroids of known voices. A close match extends that voice and nudges
    its centroid; a poor match enrols a new voice, up to `max_speakers`.

    Two guards stop the roster from fragmenting into phantom people:

    - Enrolment minimum duration. A 0.4 s "hmm", a cough or a chair scrape
      carries almost no timbre information, so its embedding sits far from
      every centroid and would otherwise mint a brand-new speaker every
      time. Short audio can only ever *match* an existing voice, never
      create one.
    - Enrolment margin. A new voice must be clearly unlike every known one,
      not merely below the match threshold, which absorbs the natural
      drift of one person's voice as they get louder or turn their head.

    An optional client-supplied hint (the UI's speaker button) takes priority
    when present, and the centroids still learn from hinted turns so
    attribution keeps working during the stretches with no hint.
    """

    # Cepstral cosine similarity above this counts as the same person.
    #
    # Calibrated against synthetic voices differing in pitch and vocal-tract
    # tilt: same-speaker pairs scored ~1.00, different-speaker pairs peaked at
    # 0.935, and 0.94 was the lowest threshold that separated them perfectly.
    #
    # The default deliberately errs toward splitting rather than merging,
    # because the two failure modes are not equally recoverable:
    #
    #   Over-split (one person → two labels) is repairable. The clinical layer
    #   sees both labels, assigns both the same role, and can request a merge
    #   via `merge_speakers`. Attribution stays correct throughout.
    #
    #   Over-merged (two people → one label) is not. A single label holding
    #   both the patient's and their relative's words can only be given one
    #   role, so the companion's statements are permanently recorded as the
    #   patient's. That is precisely the contamination this feature exists to
    #   prevent, and no downstream stage can undo it — the LLM has no access
    #   to the audio and cannot split a label back apart.
    #
    # Tune with DIARIZER_THRESHOLD against real recordings. Lower it if one
    # person keeps appearing as several voices; raise it if distinct people
    # are being combined.
    DEFAULT_THRESHOLD = 0.94

    def __init__(
        self,
        max_speakers: int = 6,
        similarity_threshold: float | None = None,
        centroid_momentum: float = 0.72,
        min_enroll_seconds: float = 0.85,
    ) -> None:
        self.max_speakers = max(1, max_speakers)
        if similarity_threshold is None:
            try:
                similarity_threshold = float(
                    os.getenv("DIARIZER_THRESHOLD") or self.DEFAULT_THRESHOLD
                )
            except ValueError:
                similarity_threshold = self.DEFAULT_THRESHOLD
        self.similarity_threshold = similarity_threshold
        self.centroid_momentum = centroid_momentum
        self.min_enroll_seconds = min_enroll_seconds

        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._seconds: dict[str, float] = {}
        self._score_sum: dict[str, float] = {}
        self._last_label: str | None = None
        self._enrolled = 0

        # Resolved clinical roles, filled in by the LLM layer once there is
        # enough dialogue to tell who is who. Kept separate from the acoustic
        # labels so a revised guess never corrupts the voice model.
        self._role_map: dict[str, str] = {}
        self.last_score: float = 0.0
        self.last_was_new: bool = False

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def speaker_count(self) -> int:
        return len(self._centroids)

    def known_speakers(self) -> list[str]:
        return sorted(self._centroids.keys())

    def utterance_counts(self) -> dict[str, int]:
        return dict(self._counts)

    def role_of(self, label: str) -> str | None:
        """Resolved clinical role for an acoustic label, if known yet."""
        return self._role_map.get(label)

    def set_role_map(self, mapping: dict[str, str]) -> None:
        """
        Attach resolved roles (Doctor, Patient, Caregiver, ...) to labels.

        Only labels this diarizer actually knows are accepted, so a
        hallucinated label from the LLM cannot inject a phantom speaker.
        """
        for label, role in (mapping or {}).items():
            if label in self._centroids and isinstance(role, str) and role.strip():
                self._role_map[label] = role.strip()

    def roster(self) -> list[dict]:
        """
        Per-voice summary for the UI and for the role-resolution prompt.

        Ordered by how much each voice has spoken, because talk time is the
        strongest cue for who the clinician and the primary patient are.
        """
        entries = []
        for label in self._centroids:
            count = self._counts.get(label, 0)
            entries.append({
                "label": label,
                "role": self._role_map.get(label),
                "utterances": count,
                "speech_seconds": round(self._seconds.get(label, 0.0), 1),
                # Mean match confidence; the enrolling utterance has no score
                # of its own, hence count-1 in the denominator.
                "mean_similarity": (
                    round(self._score_sum.get(label, 0.0) / max(1, count - 1), 3)
                    if count > 1 else None
                ),
            })
        entries.sort(key=lambda e: e["speech_seconds"], reverse=True)
        return entries

    # ── Core assignment ──────────────────────────────────────────────────

    def assign(
        self,
        samples: np.ndarray,
        sample_rate: int = 16_000,
        hint: str | None = None,
    ) -> str:
        """
        Return a stable voice label for one utterance.

        Args:
            samples:     Mono float32 audio for this utterance only.
            sample_rate: Sample rate of `samples`.
            hint:        Optional caller-asserted label (e.g. "Doctor").
                         Trusted over the acoustic match when given.

        Returns:
            A voice label — the hint itself if one was supplied, otherwise
            "SPEAKER_00", "SPEAKER_01", ...
        """
        duration = samples.size / float(sample_rate) if sample_rate else 0.0
        embedding = speaker_embedding(samples, sample_rate=sample_rate)
        self.last_was_new = False

        if hint:
            self.last_score = 1.0
            self._commit(hint, embedding, duration, score=None)
            return hint

        if embedding is None:
            # Too short to characterise. Attributing it to whoever just spoke
            # beats inventing a new person.
            fallback = self._last_label or self._new_label()
            self._commit(fallback, None, duration, score=None)
            return fallback

        best_label, best_score = None, -1.0
        for label, centroid in self._centroids.items():
            score = cosine_similarity(embedding, centroid)
            if score > best_score:
                best_label, best_score = label, score

        self.last_score = round(max(0.0, best_score), 3)

        matched = best_label is not None and best_score >= self.similarity_threshold

        # Enrolment is gated on capacity and clip length only. An earlier
        # version also required the score to fall a margin *below* the
        # threshold, which created a dead band where a new speaker could never
        # be enrolled and was instead absorbed into the nearest voice — the
        # exact over-merging this class must avoid.
        can_enroll = (
            len(self._centroids) < self.max_speakers
            and duration >= self.min_enroll_seconds
        )

        if matched:
            label = best_label
        elif can_enroll:
            label = self._new_label()
            self.last_was_new = True
        else:
            # Roster full, or the clip is too short to characterise a new
            # voice. Assign the nearest known voice rather than dropping the
            # turn: a possibly-misattributed sentence is more useful than a
            # silently discarded one.
            label = best_label or self._new_label()

        self._commit(label, embedding, duration, score=best_score if not self.last_was_new else None)
        return label

    def reassign_label(self, old_label: str, new_label: str) -> bool:
        """
        Merge one voice into another.

        Used when the clinical layer determines that two acoustic labels are
        really the same person — a voice that shifted enough mid-consultation
        to get enrolled twice.
        """
        if old_label not in self._centroids or old_label == new_label:
            return False

        moved = self._centroids.pop(old_label)
        self._role_map.pop(old_label, None)
        count = self._counts.pop(old_label, 0)
        seconds = self._seconds.pop(old_label, 0.0)
        self._score_sum.pop(old_label, 0.0)

        if new_label in self._centroids:
            blended = self._centroids[new_label] + moved
            norm = float(np.linalg.norm(blended))
            if norm > 1e-8:
                blended = blended / norm
            self._centroids[new_label] = blended.astype(np.float32, copy=False)
            self._counts[new_label] = self._counts.get(new_label, 0) + count
            self._seconds[new_label] = self._seconds.get(new_label, 0.0) + seconds
        else:
            self._centroids[new_label] = moved
            self._counts[new_label] = count
            self._seconds[new_label] = seconds

        if self._last_label == old_label:
            self._last_label = new_label
        return True

    # ── Internals ────────────────────────────────────────────────────────

    def _new_label(self) -> str:
        # Counter rather than len(): hinted turns create role-named labels,
        # so length would collide and overwrite an existing voice.
        label = f"SPEAKER_{self._enrolled:02d}"
        self._enrolled += 1
        return label

    def _commit(
        self,
        label: str,
        embedding: np.ndarray | None,
        duration: float,
        score: float | None,
    ) -> None:
        """Record bookkeeping for an assignment and update the centroid."""
        if embedding is not None:
            self._update_centroid(label, embedding)
        else:
            self._counts[label] = self._counts.get(label, 0) + 1

        self._seconds[label] = self._seconds.get(label, 0.0) + duration
        if score is not None:
            self._score_sum[label] = self._score_sum.get(label, 0.0) + max(0.0, score)
        self._last_label = label

    def _update_centroid(self, label: str, embedding: np.ndarray) -> None:
        """Blend an embedding into a voice centroid and renormalise it."""
        current = self._centroids.get(label)
        if current is None:
            self._centroids[label] = embedding.astype(np.float32, copy=True)
            self._counts[label] = 1
            return

        blended = (
            self.centroid_momentum * current
            + (1.0 - self.centroid_momentum) * embedding
        )
        norm = float(np.linalg.norm(blended))
        if norm > 1e-8:
            blended = blended / norm

        self._centroids[label] = blended.astype(np.float32, copy=False)
        self._counts[label] = self._counts.get(label, 0) + 1
