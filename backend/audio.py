"""
audio.py — Real-time audio DSP primitives (numpy only, no extra dependencies).

This module exists so the streaming pipeline never has to touch disk or spawn
ffmpeg. The browser sends raw 16-bit PCM frames over the WebSocket and every
transform needed before Whisper (resampling, voice-activity detection, speaker
embedding) happens here on plain numpy arrays.

Contents
--------
Conversion   : pcm16_to_float32 / float32_to_pcm16
Resampling   : resample_linear
Levels       : rms / rms_to_dbfs / rms_to_meter
Segmentation : VoiceActivityDetector  (energy VAD with hysteresis)
Embeddings   : log_mel_spectrogram / speaker_embedding
"""

from __future__ import annotations

import math

import numpy as np

# ── Canonical stream format ──────────────────────────────────────────────
# Whisper is trained on 16 kHz mono audio, so the whole realtime path is
# normalised to this sample rate exactly once, at ingest.
TARGET_SAMPLE_RATE = 16_000

_INT16_FULL_SCALE = 32768.0


# ══════════════════════════════════════════════════════════════════════════
# CONVERSION
# ══════════════════════════════════════════════════════════════════════════

def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """
    Decode little-endian signed 16-bit PCM bytes into float32 samples in
    the range [-1.0, 1.0].

    A trailing odd byte (possible when a WebSocket frame is split) is
    dropped rather than raising, so a malformed frame degrades one sample
    instead of killing the session.
    """
    if not raw:
        return np.zeros(0, dtype=np.float32)

    if len(raw) % 2:
        raw = raw[:-1]

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return samples / _INT16_FULL_SCALE


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Encode float32 samples in [-1.0, 1.0] back to 16-bit PCM bytes."""
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * (_INT16_FULL_SCALE - 1)).astype("<i2").tobytes()


# ══════════════════════════════════════════════════════════════════════════
# RESAMPLING
# ══════════════════════════════════════════════════════════════════════════

def resample_linear(
    samples: np.ndarray,
    src_rate: int,
    dst_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Resample mono float32 audio using linear interpolation.

    Linear interpolation is not the highest-fidelity kernel available, but it
    is allocation-cheap and fast enough to run per 100 ms chunk without
    adding measurable latency. Speech recognition accuracy is dominated by
    the acoustic model, not by resampler ripple, so this is a deliberate
    latency-over-fidelity trade.
    """
    if samples.size == 0 or src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)

    duration = samples.size / float(src_rate)
    out_count = int(round(duration * dst_rate))
    if out_count <= 0:
        return np.zeros(0, dtype=np.float32)

    src_positions = np.arange(samples.size, dtype=np.float64)
    dst_positions = np.linspace(0.0, samples.size - 1.0, out_count)
    return np.interp(dst_positions, src_positions, samples).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# LEVELS
# ══════════════════════════════════════════════════════════════════════════

def rms(samples: np.ndarray) -> float:
    """Root-mean-square amplitude of a frame (0.0 for an empty frame)."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def rms_to_dbfs(value: float) -> float:
    """
    Convert an RMS amplitude to dBFS, floored at -100 dB.

    Full scale (1.0) maps to 0 dBFS; everything in normal speech territory
    is therefore negative.
    """
    if value <= 1e-9:
        return -100.0
    return max(-100.0, 20.0 * math.log10(value))


def rms_to_meter(value: float) -> int:
    """
    Map an RMS amplitude onto a 0-100 scale for the UI level meter.

    -60 dBFS reads as 0 and 0 dBFS reads as 100, which puts conversational
    speech in the comfortable 35-75 band.
    """
    dbfs = rms_to_dbfs(value)
    return int(max(0.0, min(100.0, (dbfs + 60.0) * (100.0 / 60.0))))


# ══════════════════════════════════════════════════════════════════════════
# VOICE ACTIVITY DETECTION
# ══════════════════════════════════════════════════════════════════════════

class VoiceActivityDetector:
    """
    Energy-based VAD with hysteresis and an adaptive noise floor.

    The detector consumes fixed-size frames and reports state transitions so
    the streaming engine knows when an utterance begins and ends. Two
    mechanisms keep it stable in a noisy clinic room:

    - Adaptive noise floor: while silent, the floor tracks observed energy
      slowly, so a humming AC unit is learned as background instead of being
      transcribed forever.
    - Hysteresis: entering speech needs `entry_frames` consecutive loud
      frames and leaving it needs `hangover_ms` of quiet, which prevents a
      single consonant gap from chopping a sentence in half.

    `accept()` returns one of: "silence", "speech-start", "speech",
    "speech-end".
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        frame_ms: int = 30,
        threshold_db_above_floor: float = 9.0,
        absolute_floor_db: float = -52.0,
        entry_frames: int = 2,
        hangover_ms: int = 620,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = max(1, int(sample_rate * frame_ms / 1000))
        self.frame_ms = frame_ms
        self.threshold_db_above_floor = threshold_db_above_floor
        self.absolute_floor_db = absolute_floor_db
        self.entry_frames = max(1, entry_frames)
        self.hangover_frames = max(1, int(hangover_ms / frame_ms))

        self._noise_floor_db = absolute_floor_db
        self._loud_streak = 0
        self._quiet_streak = 0
        self._in_speech = False
        self.last_db = -100.0
        self.last_rms = 0.0

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def noise_floor_db(self) -> float:
        return self._noise_floor_db

    def reset(self) -> None:
        """Clear speech state but keep the learned noise floor."""
        self._loud_streak = 0
        self._quiet_streak = 0
        self._in_speech = False

    def accept(self, frame: np.ndarray) -> str:
        """Feed one frame and return the resulting VAD state."""
        frame_rms = rms(frame)
        frame_db = rms_to_dbfs(frame_rms)
        self.last_rms = frame_rms
        self.last_db = frame_db

        speech_threshold = max(
            self.absolute_floor_db,
            self._noise_floor_db + self.threshold_db_above_floor,
        )
        is_loud = frame_db > speech_threshold

        if not is_loud:
            # Track the ambient level only while quiet, and only upward-slowly,
            # so a long pause cannot raise the floor above real speech.
            self._noise_floor_db = (0.95 * self._noise_floor_db) + (0.05 * frame_db)

        if self._in_speech:
            if is_loud:
                self._quiet_streak = 0
                return "speech"
            self._quiet_streak += 1
            if self._quiet_streak >= self.hangover_frames:
                self._in_speech = False
                self._loud_streak = 0
                self._quiet_streak = 0
                return "speech-end"
            # Still inside the hangover window: treat as speech so trailing
            # words are not clipped off the utterance.
            return "speech"

        if is_loud:
            self._loud_streak += 1
            if self._loud_streak >= self.entry_frames:
                self._in_speech = True
                self._quiet_streak = 0
                return "speech-start"
            return "silence"

        self._loud_streak = 0
        return "silence"


# ══════════════════════════════════════════════════════════════════════════
# SPEAKER EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════

_MEL_FILTERBANK_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.power(10.0, np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Build (and cache) a triangular mel filterbank matrix."""
    key = (n_mels, n_fft, sample_rate)
    cached = _MEL_FILTERBANK_CACHE.get(key)
    if cached is not None:
        return cached

    n_bins = n_fft // 2 + 1
    # Voice-relevant band only: below 60 Hz is rumble, above 7.6 kHz is
    # mostly noise at a 16 kHz sample rate.
    low_hz, high_hz = 60.0, min(7600.0, sample_rate / 2.0)

    mel_points = np.linspace(_hz_to_mel(low_hz), _hz_to_mel(high_hz), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    filters = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if right <= left:
            # Degenerate triangle (can happen at very low n_fft) — put all
            # weight on the single centre bin so the filter still responds.
            filters[m - 1, min(center, n_bins - 1)] = 1.0
            continue
        if center > left:
            filters[m - 1, left:center] = (
                np.arange(left, center) - left
            ) / float(center - left)
        if right > center:
            filters[m - 1, center:right] = (
                right - np.arange(center, right)
            ) / float(right - center)

    _MEL_FILTERBANK_CACHE[key] = filters
    return filters


def log_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    n_mels: int = 40,
    frame_ms: int = 25,
    hop_ms: int = 10,
) -> np.ndarray:
    """
    Compute a log-mel spectrogram of shape (frames, n_mels).

    Returns an empty (0, n_mels) array when the input is shorter than one
    analysis window.
    """
    if samples.size == 0:
        return np.zeros((0, n_mels), dtype=np.float32)

    win_length = max(16, int(sample_rate * frame_ms / 1000))
    hop_length = max(1, int(sample_rate * hop_ms / 1000))
    if samples.size < win_length:
        return np.zeros((0, n_mels), dtype=np.float32)

    n_fft = 1 << (win_length - 1).bit_length()
    window = np.hanning(win_length).astype(np.float32)

    frame_count = 1 + (samples.size - win_length) // hop_length
    indices = (
        np.arange(win_length)[None, :]
        + hop_length * np.arange(frame_count)[:, None]
    )
    frames = samples[indices] * window

    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = np.square(np.abs(spectrum)).astype(np.float32)

    mel_energy = power @ _mel_filterbank(n_mels, n_fft, sample_rate).T
    return np.log(mel_energy + 1e-10).astype(np.float32)


_DCT_MATRIX_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _dct2_matrix(n_coef: int, n_mels: int) -> np.ndarray:
    """Orthonormal DCT-II matrix, used to turn log-mel bands into cepstra."""
    key = (n_coef, n_mels)
    cached = _DCT_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    k = np.arange(n_coef)[:, None]
    n = np.arange(n_mels)[None, :]
    matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    matrix *= np.sqrt(2.0 / n_mels)
    matrix[0] *= np.sqrt(0.5)

    matrix = matrix.astype(np.float32)
    _DCT_MATRIX_CACHE[key] = matrix
    return matrix


def speaker_embedding(
    samples: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    n_mels: int = 40,
    n_coef: int = 20,
) -> np.ndarray | None:
    """
    Produce a compact, L2-normalised voice-timbre embedding for one utterance.

    Built from mel-frequency cepstral coefficients rather than raw log-mel
    bands. That distinction matters for telling several people in one room
    apart:

    - The DCT decorrelates neighbouring mel bands. Adjacent bands are
      strongly correlated, so cosine similarity over raw bands is dominated
      by overall spectral tilt — which is why two clearly different voices
      can score deceptively high and get merged into one speaker.
    - Coefficient 0 is discarded. It carries frame energy, so dropping it
      makes the embedding invariant to how loudly someone speaks and how far
      they are from the microphone.
    - Cepstral means, standard deviations and first-order deltas are pooled
      together, capturing vocal-tract shape, how much it varies, and how
      quickly it moves — the last of these separates a fast, clipped speaking
      style from a slow one even when timbre is similar.

    This is deliberately a lightweight statistical embedding rather than a
    neural x-vector: it runs in well under a millisecond, needs no model
    download, and only has to separate a handful of people in one room.

    Returns None when the utterance is too short to characterise.
    """
    mels = log_mel_spectrogram(samples, sample_rate=sample_rate, n_mels=n_mels)
    if mels.shape[0] < 8:
        return None

    # (frames, n_mels) → (frames, n_coef), then drop c0 (energy).
    cepstra = mels @ _dct2_matrix(n_coef, mels.shape[1]).T
    cepstra = cepstra[:, 1:]

    parts = [cepstra.mean(axis=0), cepstra.std(axis=0)]

    if cepstra.shape[0] >= 3:
        deltas = np.diff(cepstra, axis=0)
        parts.append(deltas.std(axis=0))
    else:
        parts.append(np.zeros(cepstra.shape[1], dtype=np.float32))

    vector = np.concatenate(parts)

    if not np.all(np.isfinite(vector)):
        return None

    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return (vector / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe against zero norms."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)





###################################
