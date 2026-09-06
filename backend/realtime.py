"""
realtime.py — The streaming consultation engine.

This module turns CareNote AI from a request/response transcriber into a live
scribe. A browser opens one WebSocket, streams raw microphone PCM, and
receives a continuous flow of events: voice-activity levels, interim text,
committed utterances with speaker labels, and a clinical package (SOAP,
ICD-10, differentials, safety alerts) that is rewritten every few seconds
while the consultation is still happening.

Concurrency model
-----------------
FastAPI's event loop must never block, but every useful component here
(Whisper, Gemini, SQLite, ChromaDB) is synchronous. The engine therefore keeps
all blocking work on a dedicated thread pool and uses the loop purely for
coordination:

    WebSocket receive  ──►  feed_audio()          cheap numpy DSP, inline
                                 │
                                 ├─► partial decode   fire-and-forget, droppable
                                 │
                                 └─► utterance queue ──► _final_worker
                                                              │
                                                              ├─► live RAG index
                                                              └─► mark dirty
                                                                     │
                                              _clinical_worker ◄─────┘
                                                     │  (debounced, coalescing)
                                                     ├─► clinical_update event
                                                     └─► DB autosave

Three deliberate back-pressure rules keep latency bounded under load:

1. Interim decodes are *droppable*. If one is still running when the next is
   due, the next is skipped. Interim text is disposable by definition, so
   dropping it costs nothing and prevents a queue from forming.
2. Committed utterances go through a single-consumer queue, which guarantees
   transcript ordering regardless of how decode times vary.
3. Clinical refreshes *coalesce*. Ten utterances arriving during one Gemini
   call trigger exactly one follow-up call, not ten.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Awaitable, Callable

import numpy as np

from backend import crud
from backend.audio import (
    TARGET_SAMPLE_RATE,
    VoiceActivityDetector,
    pcm16_to_float32,
    resample_linear,
    rms,
    rms_to_dbfs,
    rms_to_meter,
)
from backend.database import SessionLocal
from backend.diarize import OnlineDiarizer
from backend.extract import (
    alert_fingerprint,
    run_advanced_clinical_extraction,
    run_incremental_extraction,
)
from backend.rag import answer_question, append_turns, drop_index, has_index
from backend.stt import transcribe_array

# ══════════════════════════════════════════════════════════════════════════
# SHARED THREAD POOL
# ══════════════════════════════════════════════════════════════════════════
#
# One pool for the whole process, not one per session. Whisper already
# serialises on its own inference lock, so more threads would only add
# contention; the extra slots exist for Gemini calls, embedding and SQLite,
# which are I/O-bound and genuinely parallel.

_WORKER_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="carenote-rt")


def shutdown_worker_pool() -> None:
    """Shut the shared pool down on application exit."""
    _WORKER_POOL.shutdown(wait=False, cancel_futures=True)


async def _in_thread(fn: Callable, *args, **kwargs):
    """Run a blocking callable on the shared pool without blocking the loop."""
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_WORKER_POOL, lambda: fn(*args, **kwargs))
    return await loop.run_in_executor(_WORKER_POOL, fn, *args)


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RealtimeConfig:
    """Tunables for one streaming session."""

    # Voice activity detection
    vad_frame_ms: int = 30
    vad_hangover_ms: int = 620
    """Silence needed to close an utterance. Long enough to survive a mid
    sentence breath, short enough that the doctor sees text land promptly."""

    # Interim decoding
    partial_interval_ms: int = 900
    """How often in-progress speech is re-decoded for on-screen interim text."""
    min_partial_ms: int = 500
    """Below this, there is not enough audio for a useful interim decode."""

    # Utterance bounds
    max_utterance_ms: int = 24_000
    """Hard cap. A monologue longer than this is split so the transcript and
    the clinical note keep updating instead of stalling until the doctor
    pauses."""
    min_utterance_ms: int = 260
    """Shorter bursts are door slams and chair scrapes, not speech."""

    # Clinical refresh
    clinical_debounce_ms: int = 2_200
    """Quiet period after the last utterance before re-deriving the note.
    Absorbs a rapid back-and-forth into a single analysis pass."""
    clinical_min_interval_ms: int = 6_000
    """Floor on the gap between two analyses, so a long monologue cannot
    trigger a Gemini call per sentence."""
    clinical_min_utterances: int = 2
    """Wait for this much dialogue before the first analysis; one line is
    rarely enough to say anything clinically useful."""

    # Multi-party attribution
    max_speakers: int = 6
    """Distinct voices tracked per encounter. Six covers a clinician, a
    patient, two accompanying relatives, a nurse, and one spare — beyond that,
    extra voices are folded into the nearest known one rather than
    fragmenting the roster."""

    # Telemetry
    level_report_ms: int = 120
    """How often audio level / VAD state is pushed for the live meter."""

    # Persistence
    autosave: bool = True

    def frame_size(self) -> int:
        return int(TARGET_SAMPLE_RATE * self.vad_frame_ms / 1000)


# ══════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SessionMetrics:
    """Observable performance counters for one session."""

    audio_seconds: float = 0.0
    utterances: int = 0
    partials: int = 0
    clinical_revisions: int = 0
    dropped_partials: int = 0
    last_final_latency_ms: float = 0.0
    last_partial_latency_ms: float = 0.0
    last_clinical_latency_ms: float = 0.0
    realtime_factor: float = 0.0
    """Decode time divided by audio duration. Below 1.0 means the engine is
    transcribing faster than the audio arrives, which is the requirement for
    a stream that never falls behind."""

    _decode_seconds: float = 0.0
    _decoded_audio_seconds: float = 0.0

    def record_final(self, audio_seconds: float, elapsed_ms: float) -> None:
        self.utterances += 1
        self.last_final_latency_ms = round(elapsed_ms, 1)
        self._decode_seconds += elapsed_ms / 1000.0
        self._decoded_audio_seconds += audio_seconds
        if self._decoded_audio_seconds > 0:
            self.realtime_factor = round(
                self._decode_seconds / self._decoded_audio_seconds, 3
            )

    def snapshot(self) -> dict:
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        data["audio_seconds"] = round(self.audio_seconds, 2)
        return data


# ══════════════════════════════════════════════════════════════════════════
# REALTIME SESSION
# ══════════════════════════════════════════════════════════════════════════

EventHandler = Callable[[dict], Awaitable[None]]


class RealtimeSession:
    """
    One live consultation.

    Lifecycle:
        session = RealtimeSession(...)
        await session.start()
        await session.feed_audio(pcm_bytes)      # repeatedly
        package = await session.finalize()
        await session.close()

    Events are pushed onto `self.events`, an asyncio.Queue the transport
    drains. Using a queue rather than calling the socket directly means a slow
    or stalled client can never block the audio pipeline.
    """

    def __init__(
        self,
        session_id: str | None = None,
        patient_id: int | None = None,
        patient_info: dict | None = None,
        config: RealtimeConfig | None = None,
        event_queue_size: int = 256,
        initial_transcript: list[dict] | None = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.patient_id = patient_id
        self.patient_info = patient_info
        self.config = config or RealtimeConfig()

        self.events: asyncio.Queue[dict] = asyncio.Queue(maxsize=event_queue_size)
        self.metrics = SessionMetrics()

        # ── Transcript state ─────────────────────────────────────────────
        # A non-empty `initial_transcript` means this is a reconnect: the
        # network dropped mid-encounter and the client is resuming. Seeding
        # the prior turns is what makes the final note cover the whole visit
        # rather than only the part after the drop.
        self.transcript: list[dict] = list(initial_transcript or [])
        self.resumed = bool(initial_transcript)

        # ── Multi-party attribution state ────────────────────────────────
        # Acoustic label → clinical role, plus the relationship of any
        # caregiver ("son", "spouse") and who supplied the history. Held on the
        # session because roles are refined over the course of the encounter.
        self.speaker_roles: dict[str, str] = {}
        self.speaker_relationships: dict[str, str] = {}
        self.history_source: str = "Patient"
        self.clinical_package: dict | None = None
        self.clinical_revision = 0
        self.final_package: dict | None = None
        self.consultation_id: int | None = None

        # ── Audio / VAD state ────────────────────────────────────────────
        self._vad = VoiceActivityDetector(
            frame_ms=self.config.vad_frame_ms,
            hangover_ms=self.config.vad_hangover_ms,
        )
        self._frame_size = self.config.frame_size()
        self._residue = np.zeros(0, dtype=np.float32)
        self._utterance_chunks: list[np.ndarray] = []
        self._utterance_samples = 0
        self._utterance_start_sec = 0.0
        self._speaker_hint: str | None = None

        # Continue the timeline where the previous connection left off so
        # segment timestamps stay monotonic across a reconnect.
        resume_offset = 0.0
        if self.transcript:
            resume_offset = float(self.transcript[-1].get("end") or 0.0)
        self._total_samples = int(resume_offset * TARGET_SAMPLE_RATE)

        # ── Scheduling state ─────────────────────────────────────────────
        self._diarizer = OnlineDiarizer(max_speakers=self.config.max_speakers)
        self._utterance_queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._final_task: asyncio.Task | None = None
        self._clinical_task: asyncio.Task | None = None
        self._partial_inflight = False
        self._samples_since_partial = 0
        self._last_level_push = 0.0
        self._clinical_dirty = asyncio.Event()
        self._last_clinical_at = 0.0
        self._indexed_turns = 0
        self._seen_alerts: set[str] = set()

        self._started_at = time.monotonic()
        self._closed = False
        self._accepting_audio = True

    # ══════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Spin up the background workers and announce the session."""
        self._final_task = asyncio.create_task(
            self._final_worker(), name=f"rt-final-{self.session_id}"
        )
        self._clinical_task = asyncio.create_task(
            self._clinical_worker(), name=f"rt-clinical-{self.session_id}"
        )

        if self.config.autosave:
            # Create the DB row immediately so a session that crashes
            # mid-consultation still leaves a recoverable record.
            try:
                self.consultation_id = await _in_thread(
                    _db_create_consultation, self.session_id, self.patient_id
                )
            except Exception as e:
                print(f"[realtime] Could not pre-create consultation row: {e}")

        if self.transcript:
            # Resumed session: re-index the recovered turns so the copilot
            # can still search them, and refresh the note right away.
            asyncio.create_task(self._index_new_turns())
            self._clinical_dirty.set()

        await self._emit("session_started", {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "patient_name": (self.patient_info or {}).get("name"),
            "sample_rate": TARGET_SAMPLE_RATE,
            "resumed": self.resumed,
            "transcript": list(self.transcript),
            "config": {
                "partial_interval_ms": self.config.partial_interval_ms,
                "clinical_debounce_ms": self.config.clinical_debounce_ms,
                "vad_hangover_ms": self.config.vad_hangover_ms,
            },
        })

    async def close(self) -> None:
        """Cancel workers and release per-session resources. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._accepting_audio = False

        for task in (self._final_task, self._clinical_task):
            if task and not task.done():
                task.cancel()
        for task in (self._final_task, self._clinical_task):
            if task:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # The Chroma client is in-memory and process-lifetime, so a live
        # index would otherwise leak for every consultation ever held.
        try:
            await _in_thread(drop_index, self.session_id)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # AUDIO INGEST
    # ══════════════════════════════════════════════════════════════════════

    async def feed_audio(self, raw: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
        """
        Accept one chunk of 16-bit PCM from the client.

        Everything here is O(chunk) numpy work measured in microseconds, so it
        runs inline on the event loop. The only work handed to threads is the
        actual decoding, which is scheduled but not awaited.
        """
        if not self._accepting_audio or not raw:
            return

        samples = pcm16_to_float32(raw)
        if samples.size == 0:
            return
        if sample_rate != TARGET_SAMPLE_RATE:
            samples = resample_linear(samples, sample_rate, TARGET_SAMPLE_RATE)

        self.metrics.audio_seconds += samples.size / TARGET_SAMPLE_RATE
        await self._push_level(samples)

        # Join with whatever did not fill a frame last time, then consume
        # whole VAD frames only.
        buffer = (
            samples if self._residue.size == 0
            else np.concatenate([self._residue, samples])
        )
        frame_count = buffer.size // self._frame_size
        self._residue = buffer[frame_count * self._frame_size:].copy()

        for i in range(frame_count):
            frame = buffer[i * self._frame_size:(i + 1) * self._frame_size]
            await self._consume_frame(frame)

    async def _consume_frame(self, frame: np.ndarray) -> None:
        """Run VAD on one frame and drive the utterance state machine."""
        state = self._vad.accept(frame)
        self._total_samples += frame.size

        if state == "speech-start":
            self._begin_utterance(frame)
            await self._emit("speech_state", {"speaking": True})
            return

        if state == "speech":
            self._utterance_chunks.append(frame)
            self._utterance_samples += frame.size
            self._samples_since_partial += frame.size

            if self._utterance_samples >= self._ms_to_samples(self.config.max_utterance_ms):
                # Split a long monologue rather than letting the transcript
                # freeze until the speaker finally pauses.
                await self._commit_utterance(forced=True)
                return

            if self._should_run_partial():
                self._schedule_partial()
            return

        if state == "speech-end":
            self._utterance_chunks.append(frame)
            self._utterance_samples += frame.size
            await self._emit("speech_state", {"speaking": False})
            await self._commit_utterance(forced=False)

    def _begin_utterance(self, frame: np.ndarray) -> None:
        self._utterance_chunks = [frame]
        self._utterance_samples = frame.size
        self._samples_since_partial = 0
        self._utterance_start_sec = max(
            0.0, (self._total_samples - frame.size) / TARGET_SAMPLE_RATE
        )

    def _ms_to_samples(self, ms: int) -> int:
        return int(TARGET_SAMPLE_RATE * ms / 1000)

    def _should_run_partial(self) -> bool:
        if self._partial_inflight:
            return False
        if self._utterance_samples < self._ms_to_samples(self.config.min_partial_ms):
            return False
        return self._samples_since_partial >= self._ms_to_samples(
            self.config.partial_interval_ms
        )

    async def _push_level(self, samples: np.ndarray) -> None:
        """Throttled audio-level telemetry for the UI meter."""
        now = time.monotonic()
        if (now - self._last_level_push) * 1000 < self.config.level_report_ms:
            return
        self._last_level_push = now

        amplitude = rms(samples)
        await self._emit("level", {
            "rms": round(amplitude, 5),
            "dbfs": round(rms_to_dbfs(amplitude), 1),
            "meter": rms_to_meter(amplitude),
            "speaking": self._vad.in_speech,
            "noise_floor_db": round(self._vad.noise_floor_db, 1),
            "elapsed_sec": round(self.metrics.audio_seconds, 1),
        }, droppable=True)

    # ══════════════════════════════════════════════════════════════════════
    # INTERIM DECODING
    # ══════════════════════════════════════════════════════════════════════

    def _schedule_partial(self) -> None:
        """Kick off a droppable interim decode of the in-progress utterance."""
        self._partial_inflight = True
        self._samples_since_partial = 0
        audio = np.concatenate(self._utterance_chunks)
        asyncio.create_task(self._run_partial(audio))

    async def _run_partial(self, audio: np.ndarray) -> None:
        started = time.monotonic()
        try:
            text = await _in_thread(
                _decode_partial, audio, self._decoder_context()
            )
            if not text:
                return
            self.metrics.partials += 1
            self.metrics.last_partial_latency_ms = round(
                (time.monotonic() - started) * 1000, 1
            )
            await self._emit("partial_transcript", {
                "text": text,
                "speaker": self._speaker_hint or "…",
                "start": round(self._utterance_start_sec, 2),
                "latency_ms": self.metrics.last_partial_latency_ms,
            }, droppable=True)
        except Exception as e:
            self.metrics.dropped_partials += 1
            print(f"[realtime] Interim decode failed: {e}")
        finally:
            self._partial_inflight = False

    def _decoder_context(self) -> str:
        """
        Tail of the committed transcript, used as Whisper's initial prompt.

        Feeding recent context back into the decoder is what keeps drug names
        and clinical terms spelled consistently across utterance boundaries,
        instead of each utterance being decoded as if it were the first thing
        ever said.
        """
        if not self.transcript:
            return ""
        tail = " ".join(t.get("text", "") for t in self.transcript[-4:])
        return tail[-420:]

    # ══════════════════════════════════════════════════════════════════════
    # UTTERANCE COMMIT
    # ══════════════════════════════════════════════════════════════════════

    async def _commit_utterance(self, forced: bool) -> None:
        """
        Close the current utterance and queue it for accurate decoding.

        `forced` means the length cap fired rather than the VAD, so the
        speaker is still talking and VAD state must be preserved.
        """
        if not self._utterance_chunks:
            return

        audio = np.concatenate(self._utterance_chunks)
        start_sec = self._utterance_start_sec

        self._utterance_chunks = []
        self._utterance_samples = 0
        self._samples_since_partial = 0

        if forced:
            # Continue the same speech run: the next frames belong to a new
            # utterance that starts immediately where this one ended.
            self._utterance_start_sec = start_sec + audio.size / TARGET_SAMPLE_RATE

        duration_ms = (audio.size / TARGET_SAMPLE_RATE) * 1000
        if duration_ms < self.config.min_utterance_ms:
            return

        await self._utterance_queue.put({
            "audio": audio,
            "start": start_sec,
            "end": start_sec + audio.size / TARGET_SAMPLE_RATE,
            "hint": self._speaker_hint,
        })

    async def _final_worker(self) -> None:
        """
        Single consumer of the utterance queue.

        Being single-consumer is the point: it makes transcript order
        deterministic even when one utterance takes far longer to decode than
        the next.
        """
        while True:
            item = await self._utterance_queue.get()
            if item is None:  # shutdown sentinel
                self._utterance_queue.task_done()
                return
            try:
                await self._process_final(item)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[realtime] Utterance processing failed: {e}")
                await self._emit("warning", {
                    "message": f"An utterance could not be transcribed: {e}"
                })
            finally:
                self._utterance_queue.task_done()

    async def _process_final(self, item: dict) -> None:
        audio: np.ndarray = item["audio"]
        started = time.monotonic()

        text = await _in_thread(_decode_final, audio, self._decoder_context())
        elapsed_ms = (time.monotonic() - started) * 1000

        if not text:
            return

        speaker = await _in_thread(
            self._diarizer.assign, audio, TARGET_SAMPLE_RATE, item.get("hint")
        )
        voice_is_new = self._diarizer.last_was_new

        turn = {
            "index": len(self.transcript),
            "speaker": speaker,
            # Resolved clinical role, once the LLM has worked out who is who.
            # Null on the first turns from a new voice, and backfilled later.
            "role": self._diarizer.role_of(speaker),
            "text": text,
            "start": round(item["start"], 2),
            "end": round(item["end"], 2),
        }
        self.transcript.append(turn)
        self.metrics.record_final(audio.size / TARGET_SAMPLE_RATE, elapsed_ms)

        await self._emit("final_utterance", {
            **turn,
            "latency_ms": round(elapsed_ms, 1),
            "speaker_count": self._diarizer.speaker_count,
            "voice_similarity": self._diarizer.last_score,
            "word_count": sum(len(t["text"].split()) for t in self.transcript),
        })

        if voice_is_new:
            # A third or fourth person just joined the conversation. Surface it
            # immediately: the clinician should know a companion is being
            # recorded as a separate voice, and roles are still unresolved.
            await self._emit("speaker_joined", {
                "label": speaker,
                "speaker_count": self._diarizer.speaker_count,
                "roster": self._diarizer.roster(),
            })

        asyncio.create_task(self._index_new_turns())
        self._clinical_dirty.set()

    async def _index_new_turns(self) -> None:
        """Keep the live RAG index current so the copilot can answer mid-visit."""
        pending = self.transcript[self._indexed_turns:]
        if not pending:
            return
        start_index = self._indexed_turns
        self._indexed_turns = len(self.transcript)
        try:
            await _in_thread(append_turns, self.session_id, list(pending), start_index)
        except Exception as e:
            # Roll back so the next commit retries these turns.
            self._indexed_turns = start_index
            print(f"[realtime] Live indexing failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # LIVE CLINICAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════

    async def _clinical_worker(self) -> None:
        """
        Debounced, coalescing loop that re-derives the clinical package.

        Waiting on an Event rather than reading a queue is what produces the
        coalescing: any number of utterances arriving while an analysis is in
        flight collapse into a single "dirty" flag and therefore a single
        follow-up analysis.
        """
        while True:
            await self._clinical_dirty.wait()

            # Settle: keep extending the wait while new speech keeps arriving,
            # so a rapid exchange is analysed once at the end rather than
            # repeatedly in the middle.
            while True:
                seen = len(self.transcript)
                await asyncio.sleep(self.config.clinical_debounce_ms / 1000)
                if len(self.transcript) != seen:
                    continue  # more speech landed; keep waiting
                if seen >= self.config.clinical_min_utterances:
                    break
                # Not enough dialogue to analyse yet. Sleep until the next
                # utterance re-signals, instead of spinning on the timer.
                self._clinical_dirty.clear()
                await self._clinical_dirty.wait()

            since_last = (time.monotonic() - self._last_clinical_at) * 1000
            if self._last_clinical_at and since_last < self.config.clinical_min_interval_ms:
                await asyncio.sleep(
                    (self.config.clinical_min_interval_ms - since_last) / 1000
                )

            self._clinical_dirty.clear()
            try:
                await self._refresh_clinical()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[realtime] Live clinical analysis failed: {e}")
                await self._emit("warning", {
                    "message": f"Live note refresh failed, will retry: {e}"
                })

    async def _refresh_clinical(self) -> None:
        turns = [
            {"speaker": t["speaker"], "text": t["text"]} for t in self.transcript
        ]
        if not turns:
            return

        await self._emit("status", {"stage": "analyzing", "message": "Updating clinical note…"})

        started = time.monotonic()
        package = await _in_thread(
            run_incremental_extraction,
            turns,
            self.patient_info,
            self.metrics.audio_seconds,
            self._diarizer.roster(),
        )
        elapsed_ms = (time.monotonic() - started) * 1000

        # Role resolution rides along on the same call that produces the note,
        # so identifying a third speaker costs no extra latency or quota.
        await self._apply_speaker_roles(package)

        self._last_clinical_at = time.monotonic()
        self.clinical_revision += 1
        self.clinical_package = package
        self.metrics.clinical_revisions = self.clinical_revision
        self.metrics.last_clinical_latency_ms = round(elapsed_ms, 1)

        await self._emit("clinical_update", {
            **package,
            "revision": self.clinical_revision,
            "based_on_utterances": len(self.transcript),
            "latency_ms": self.metrics.last_clinical_latency_ms,
            "metrics": self.metrics.snapshot(),
        })

        await self._push_new_alerts(package.get("safety_alerts") or [])

        if self.config.autosave:
            asyncio.create_task(self._autosave(package))

    async def _apply_speaker_roles(self, package: dict) -> None:
        """
        Adopt the clinical layer's view of who each voice is.

        Role resolution improves as the conversation develops: two turns in, a
        companion who answers on the patient's behalf is genuinely hard to tell
        from the patient, and the early guess should be allowed to correct
        itself. So roles are re-derived on every refresh and any change is
        pushed to the client, which relabels the turns already on screen rather
        than leaving a stale "Patient" badge on a caregiver's words.
        """
        # Fold duplicate voices first: one person whose voice drifted enough to
        # be enrolled twice should not occupy two roster slots.
        for pair in (package.get("merge_speakers") or []):
            try:
                source, target = pair[0], pair[1]
            except (TypeError, IndexError):
                continue
            if await _in_thread(self._diarizer.reassign_label, source, target):
                for turn in self.transcript:
                    if turn.get("speaker") == source:
                        turn["speaker"] = target

        roles = package.get("speaker_roles") or {}
        relationships = package.get("speaker_relationships") or {}
        if not roles and not relationships:
            return

        previous = dict(self.speaker_roles)
        self._diarizer.set_role_map(roles)

        # Read roles back from the diarizer rather than trusting the package:
        # it drops labels that were never actually enrolled.
        self.speaker_roles = {
            label: role
            for label in self._diarizer.known_speakers()
            if (role := self._diarizer.role_of(label))
        }
        self.speaker_relationships.update({
            str(k): str(v) for k, v in relationships.items() if v
        })
        self.history_source = package.get("history_source") or self.history_source

        # Backfill turns recorded before their speaker's role was known.
        for turn in self.transcript:
            resolved = self.speaker_roles.get(turn.get("speaker"))
            if resolved:
                turn["role"] = resolved

        if self.speaker_roles == previous:
            return

        await self._emit("speaker_roles", {
            "roles": dict(self.speaker_roles),
            "relationships": dict(self.speaker_relationships),
            "history_source": self.history_source,
            "roster": self._diarizer.roster(),
            "excluded_mentions": package.get("excluded_mentions") or [],
        })

    async def _push_new_alerts(self, alerts: list[dict]) -> None:
        """
        Emit a dedicated event only for alerts not seen before.

        The clinical package always carries the complete alert list, so
        without fingerprinting the UI would re-raise the same allergy warning
        on every revision and the doctor would learn to ignore it.
        """
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            fingerprint = alert_fingerprint(alert)
            if fingerprint in self._seen_alerts:
                continue
            self._seen_alerts.add(fingerprint)
            await self._emit("safety_alert", {
                "category": alert.get("category", "warning"),
                "severity": alert.get("severity", "medium"),
                "message": alert.get("message", ""),
                "at_second": round(self.metrics.audio_seconds, 1),
            })

    async def _autosave(self, package: dict) -> None:
        """Persist the in-progress transcript and note so nothing is lost."""
        try:
            await _in_thread(
                _db_autosave,
                self.session_id,
                self.patient_id,
                list(self.transcript),
                package,
                "live",
            )
        except Exception as e:
            print(f"[realtime] Autosave failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # CLIENT COMMANDS
    # ══════════════════════════════════════════════════════════════════════

    def set_speaker_hint(self, speaker: str | None) -> None:
        """
        Set or clear the caller-asserted active speaker.

        The UI's Doctor/Patient toggle is far more reliable than any acoustic
        guess, so when it is set the diarizer defers to it — while still
        learning both voices for the stretches where it is not.
        """
        if speaker in (None, "", "auto"):
            self._speaker_hint = None
        else:
            self._speaker_hint = str(speaker)

    async def attach_patient(self, patient_id: int | None, patient_info: dict | None) -> None:
        """Link (or relink) a patient mid-session and re-check safety at once."""
        self.patient_id = patient_id
        self.patient_info = patient_info
        # A newly linked allergy list can invalidate an already-issued
        # prescription, so re-run analysis instead of waiting for more speech.
        self._seen_alerts.clear()
        if self.transcript:
            self._clinical_dirty.set()

    async def ask(self, question: str) -> str:
        """Answer a question about the consultation so far, live."""
        await self._index_new_turns()
        if not await _in_thread(has_index, self.session_id):
            return "Nothing has been transcribed yet in this consultation."
        return await _in_thread(answer_question, self.session_id, question)

    async def flush(self) -> None:
        """Commit any buffered speech and wait for the decode queue to drain."""
        self._vad.reset()
        await self._commit_utterance(forced=False)
        await self._utterance_queue.join()

    async def finalize(self) -> dict:
        """
        End the consultation and produce the definitive clinical package.

        The live packages are provisional by construction — they are prompted
        to leave unreached sections pending. Finalisation therefore runs the
        full non-incremental extraction over the complete transcript so the
        stored record is a finished document, not the last live snapshot.
        """
        self._accepting_audio = False
        await self._emit("status", {"stage": "finalizing", "message": "Completing clinical note…"})

        await self.flush()

        if not self.transcript:
            package = {
                "session_id": self.session_id,
                "consultation_id": self.consultation_id,
                "role_labeled_transcript": [],
                "soap_note": "No speech was captured in this session.",
                "extraction": {},
                "icd10_codes": [],
                "differential_diagnosis": [],
                "safety_alerts": [],
                "patient_instructions": "",
                "action_summary": [],
                "speaker_roles": {},
                "speaker_relationships": {},
                "history_source": "Patient",
                "excluded_mentions": [],
                "speaker_roster": [],
                "metrics": self.metrics.snapshot(),
            }
            self.final_package = package
            await self._emit("finalized", package)
            return package

        turns = [{"speaker": t["speaker"], "text": t["text"]} for t in self.transcript]

        started = time.monotonic()
        try:
            result = await _in_thread(
                run_advanced_clinical_extraction,
                turns,
                self.patient_info,
                self._diarizer.roster(),
            )
        except Exception as e:
            print(f"[realtime] Final extraction failed: {e}")
            # Never lose the encounter: fall back to the last live package.
            result = self.clinical_package or {}

        elapsed_sec = time.monotonic() - started

        await self._apply_speaker_roles(result)

        role_labeled = result.get("role_labeled_transcript") or turns
        package = {
            "session_id": self.session_id,
            "consultation_id": self.consultation_id,
            "role_labeled_transcript": role_labeled,
            "soap_note": result.get("soap_note", ""),
            "extraction": result.get("extraction", {}),
            "icd10_codes": result.get("icd10_codes", []),
            "differential_diagnosis": result.get("differential_diagnosis", []),
            "safety_alerts": result.get("safety_alerts", []),
            "patient_instructions": result.get("patient_instructions", ""),
            "action_summary": result.get("action_summary", []),
            # Who was in the room and who gave the history. Persisted with the
            # encounter so the record shows whether history was first-hand.
            "speaker_roles": dict(self.speaker_roles),
            "speaker_relationships": dict(self.speaker_relationships),
            "history_source": self.history_source,
            "excluded_mentions": result.get("excluded_mentions", []),
            "speaker_roster": self._diarizer.roster(),
            "duration_sec": round(elapsed_sec, 2),
            "metrics": self.metrics.snapshot(),
        }

        if self.config.autosave:
            try:
                await _in_thread(
                    _db_autosave,
                    self.session_id,
                    self.patient_id,
                    list(self.transcript),
                    package,
                    "completed",
                )
            except Exception as e:
                print(f"[realtime] Final save failed: {e}")

        self.final_package = package
        await self._push_new_alerts(package.get("safety_alerts") or [])
        await self._emit("finalized", package)
        return package

    # ══════════════════════════════════════════════════════════════════════
    # EVENT EMISSION
    # ══════════════════════════════════════════════════════════════════════

    async def _emit(self, event_type: str, payload: dict, droppable: bool = False) -> None:
        """
        Queue an event for the transport.

        `droppable` marks high-frequency telemetry (audio levels, interim
        text). When the client cannot keep up, dropping those is strictly
        better than applying back-pressure to the audio pipeline; transcript
        and clinical events are never dropped.
        """
        message = {
            "type": event_type,
            "session_id": self.session_id,
            "t": round(time.monotonic() - self._started_at, 3),
            **payload,
        }
        if droppable:
            try:
                self.events.put_nowait(message)
            except asyncio.QueueFull:
                pass
            return
        await self.events.put(message)

    def status(self) -> dict:
        """Point-in-time summary, used by the REST introspection endpoint."""
        return {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "utterances": len(self.transcript),
            "clinical_revision": self.clinical_revision,
            "speakers": self._diarizer.known_speakers(),
            "speaker_roles": dict(self.speaker_roles),
            "speaker_roster": self._diarizer.roster(),
            "history_source": self.history_source,
            "speaker_hint": self._speaker_hint,
            "accepting_audio": self._accepting_audio,
            "finalized": self.final_package is not None,
            "uptime_sec": round(time.monotonic() - self._started_at, 1),
            "metrics": self.metrics.snapshot(),
        }


# ══════════════════════════════════════════════════════════════════════════
# BLOCKING HELPERS (executed on the worker pool)
# ══════════════════════════════════════════════════════════════════════════

def _decode_partial(audio: np.ndarray, context: str) -> str:
    segments = transcribe_array(audio, partial=True, initial_prompt=context or None)
    return " ".join(s["text"] for s in segments).strip()


def _decode_final(audio: np.ndarray, context: str) -> str:
    segments = transcribe_array(audio, partial=False, initial_prompt=context or None)
    return " ".join(s["text"] for s in segments).strip()


def _db_create_consultation(session_id: str, patient_id: int | None) -> int | None:
    db = SessionLocal()
    try:
        existing = crud.get_consultation_by_session(db, session_id)
        if existing:
            return existing.id
        record = crud.create_consultation(
            db,
            session_id=session_id,
            audio_filename="live-stream",
            patient_id=patient_id,
            transcript=[],
        )
        return record.id
    finally:
        db.close()


def _db_autosave(
    session_id: str,
    patient_id: int | None,
    transcript: list[dict],
    package: dict,
    stage: str,
) -> None:
    """
    Write the current transcript and clinical package to SQLite.

    Called from a worker thread with its own Session, which is why
    `database.py` configures the SQLite engine with check_same_thread=False.
    """
    db = SessionLocal()
    try:
        record = crud.get_consultation_by_session(db, session_id)
        if record is None:
            record = crud.create_consultation(
                db,
                session_id=session_id,
                audio_filename="live-stream",
                patient_id=patient_id,
                transcript=transcript,
            )
        else:
            record.transcript_json = json.dumps(transcript)
            if patient_id and not record.patient_id:
                record.patient_id = patient_id
            db.commit()

        extraction = dict(package.get("extraction") or {})
        extraction.update({
            "icd10_codes": package.get("icd10_codes", []),
            "differential_diagnosis": package.get("differential_diagnosis", []),
            "safety_alerts": package.get("safety_alerts", []),
            "patient_instructions": package.get("patient_instructions", ""),
        })

        crud.update_consultation_extraction(
            db,
            session_id=session_id,
            soap_note=package.get("soap_note", ""),
            extraction=extraction,
            action_items=package.get("action_summary", []),
        )

        if stage == "live":
            # update_consultation_extraction marks the row "completed" once a
            # SOAP note exists. An in-progress encounter is not completed, so
            # correct the status back until finalisation says otherwise.
            record = crud.get_consultation_by_session(db, session_id)
            if record is not None:
                record.status = "live"
                db.commit()
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════════════
# SESSION REGISTRY
# ══════════════════════════════════════════════════════════════════════════

class SessionRegistry:
    """Tracks every open streaming session in the process."""

    def __init__(self) -> None:
        self._sessions: dict[str, RealtimeSession] = {}
        self._lock = asyncio.Lock()

    async def create(self, **kwargs) -> RealtimeSession:
        session = RealtimeSession(**kwargs)
        async with self._lock:
            self._sessions[session.session_id] = session
        await session.start()
        return session

    def get(self, session_id: str) -> RealtimeSession | None:
        return self._sessions.get(session_id)

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            await session.close()

    def active_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def status(self) -> list[dict]:
        return [s.status() for s in self._sessions.values()]

    async def close_all(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                await session.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD EVENT BUS
# ══════════════════════════════════════════════════════════════════════════

class EventBus:
    """
    Fan-out of server-wide events to any number of dashboard subscribers.

    Each subscriber gets its own bounded queue and slow subscribers are
    dropped rather than allowed to slow the publisher — a stalled dashboard
    tab must never affect a consultation in progress.
    """

    def __init__(self, queue_size: int = 64) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._queue_size = queue_size

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, event_type: str, payload: dict | None = None) -> None:
        """Non-blocking broadcast; safe to call from any coroutine."""
        message = {"type": event_type, "ts": time.time(), **(payload or {})}
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass


# Process-wide singletons used by the FastAPI layer.
registry = SessionRegistry()
event_bus = EventBus()
