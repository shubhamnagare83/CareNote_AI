"""
main.py — FastAPI application for the CareNote AI Clinical Scribe.

Endpoints:
  POST /consult/upload         — Upload audio, transcribe + diarize
  GET  /consult/{id}/note      — Generate clinical extraction + SOAP note
  GET  /consult/{id}/actions   — Get action items from extraction
  POST /consult/{id}/ask       — RAG Q&A over transcript

  # Patient CRUD
  POST   /patients             — Create patient
  GET    /patients             — List all patients
  GET    /patients/{id}        — Get single patient
  PUT    /patients/{id}        — Update patient
  DELETE /patients/{id}        — Delete patient

  # Consultation CRUD
  GET    /consultations        — List all consultations
  GET    /consultations/{id}   — Get consultation detail
  DELETE /consultations/{id}   — Delete consultation

  # Dashboard
  GET    /dashboard/stats      — Dashboard statistics

  # Health
  GET    /health               — Health check
"""

import asyncio
import json
import os
import uuid
import tempfile
import shutil

from dotenv import load_dotenv

# Load .env before any module that reads env vars
load_dotenv()

import torch
from fastapi import (
    FastAPI, UploadFile, File, HTTPException, Depends, Query,
    WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.stt import transcribe
from backend.diarize import diarize, merge_transcript_and_diarization
from backend.extract import run_full_extraction, run_advanced_clinical_extraction
from backend.rag import build_index, answer_question
from backend.realtime import (
    RealtimeConfig, RealtimeSession, event_bus, registry, shutdown_worker_pool,
)
from backend.schemas import (
    AskRequest, AskResponse,
    PatientCreate, PatientUpdate, PatientResponse, ConsultationResponse,
    FastProcessRequest,
)
from backend.database import get_db, init_db, SessionLocal
from backend import crud

# ── App setup ────────────────────────────────────────────────────────────

app = FastAPI(
    title="CareNote AI — Clinical Scribe",
    description="Ambient AI clinical scribe: transcribe, diarize, extract, and query doctor-patient consultations.",
    version="0.2.0",
)

# CORS: allow all origins (hackathon prototype)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize database on startup ──────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    """
    Create database tables and warm the model cache.

    Model weights load in the background rather than blocking startup, so the
    server accepts connections immediately. Warming matters for the realtime
    path: without it the first utterance of the day would wait several
    seconds for Whisper to load, which is exactly the moment a clinician is
    deciding whether live transcription actually works.
    """
    init_db()

    if os.getenv("PRELOAD_MODELS", "1") not in {"0", "false", "False"}:
        asyncio.create_task(_warm_models())


async def _warm_models() -> None:
    from backend.rag import preload_embedding_model
    from backend.stt import preload_models

    try:
        await asyncio.to_thread(preload_models)
        await asyncio.to_thread(preload_embedding_model)
        print("[startup] Realtime models warmed and ready.")
    except Exception as e:
        print(f"[startup] Model warm-up skipped: {e}")


@app.on_event("shutdown")
async def on_shutdown():
    """Close every live stream and release the shared thread pool."""
    await registry.close_all()
    shutdown_worker_pool()


# ── In-memory session store ──────────────────────────────────────────────
# Key: session_id (str)
# Value: dict with keys:
#   "audio_path": str (path to saved temp file)
#   "transcript": list[dict]          — raw STT segments
#   "diarization": list[dict]         — raw diarization turns
#   "merged_transcript": list[dict]   — speaker-labeled segments
#   "extraction_result": dict | None  — output of run_full_extraction
#   "rag_indexed": bool               — whether RAG index has been built

SESSIONS: dict[str, dict] = {}


# ══════════════════════════════════════════════════════════════════════════
# PATIENT CRUD ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.post("/patients", response_model=PatientResponse)
async def create_patient(data: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient record."""
    patient = crud.create_patient(db, data)
    event_bus.publish("patient_changed", {"action": "created", "patient_id": patient.id})
    return _patient_to_response(db, patient)


@app.get("/patients", response_model=list[PatientResponse])
async def list_patients(
    search: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List all patients. Supports search by name and pagination."""
    if search:
        patients = crud.search_patients(db, search)
    else:
        patients = crud.get_patients(db, skip=skip, limit=limit)
    return [_patient_to_response(db, p) for p in patients]


@app.get("/patients/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: int, db: Session = Depends(get_db)):
    """Get a single patient by ID."""
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return _patient_to_response(db, patient)


@app.put("/patients/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: int, data: PatientUpdate, db: Session = Depends(get_db)
):
    """Update an existing patient."""
    patient = crud.update_patient(db, patient_id, data)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return _patient_to_response(db, patient)


@app.delete("/patients/{patient_id}")
async def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    """Delete a patient and all associated consultations."""
    deleted = crud.delete_patient(db, patient_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Patient not found")
    event_bus.publish("patient_changed", {"action": "deleted", "patient_id": patient_id})
    return {"message": "Patient deleted successfully"}


def _patient_to_response(db: Session, patient) -> dict:
    """Convert Patient ORM to response dict with consultation count."""
    return {
        "id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "phone": patient.phone,
        "email": patient.email,
        "blood_group": patient.blood_group,
        "medical_history": patient.medical_history,
        "allergies": patient.allergies,
        "created_at": patient.created_at,
        "updated_at": patient.updated_at,
        "consultation_count": len(patient.consultations) if patient.consultations else 0,
    }


# ══════════════════════════════════════════════════════════════════════════
# CONSULTATION CRUD ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/consultations", response_model=list[ConsultationResponse])
async def list_consultations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List all consultations with pagination."""
    consultations = crud.get_consultations(db, skip=skip, limit=limit)
    return [_consultation_to_response(c) for c in consultations]


@app.get("/consultations/{consultation_id}")
async def get_consultation(consultation_id: int, db: Session = Depends(get_db)):
    """Get full consultation details including transcript and notes."""
    consultation = crud.get_consultation(db, consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return {
        "id": consultation.id,
        "session_id": consultation.session_id,
        "patient_id": consultation.patient_id,
        "patient_name": consultation.patient.name if consultation.patient else None,
        "audio_filename": consultation.audio_filename,
        "status": consultation.status,
        "created_at": consultation.created_at,
        "transcript": json.loads(consultation.transcript_json) if consultation.transcript_json else [],
        "soap_note": consultation.soap_note,
        "extraction": json.loads(consultation.extraction_json) if consultation.extraction_json else None,
        "action_items": json.loads(consultation.action_items_json) if consultation.action_items_json else [],
    }


@app.get("/patients/{patient_id}/consultations")
async def get_patient_consultations(patient_id: int, db: Session = Depends(get_db)):
    """Get all consultations for a specific patient."""
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    consultations = crud.get_consultations_by_patient(db, patient_id)
    return [_consultation_to_response(c) for c in consultations]


@app.delete("/consultations/{consultation_id}")
async def delete_consultation(consultation_id: int, db: Session = Depends(get_db)):
    """Delete a consultation."""
    deleted = crud.delete_consultation(db, consultation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Consultation not found")
    event_bus.publish("consultation_ended", {"action": "deleted", "consultation_id": consultation_id})
    return {"message": "Consultation deleted successfully"}


def _consultation_to_response(consultation) -> dict:
    """Convert Consultation ORM to response dict."""
    return {
        "id": consultation.id,
        "session_id": consultation.session_id,
        "patient_id": consultation.patient_id,
        "patient_name": consultation.patient.name if consultation.patient else None,
        "audio_filename": consultation.audio_filename,
        "status": consultation.status,
        "created_at": consultation.created_at,
        "has_transcript": bool(consultation.transcript_json),
        "has_soap_note": bool(consultation.soap_note),
        "has_extraction": bool(consultation.extraction_json),
    }


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD STATS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/dashboard/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics."""
    total_patients = crud.get_patient_count(db)
    total_consultations = crud.get_consultation_count(db)
    today_consultations = crud.get_today_consultation_count(db)

    # Recent consultations (last 5)
    recent = crud.get_consultations(db, skip=0, limit=5)
    recent_list = [_consultation_to_response(c) for c in recent]

    # Count completed vs pending
    all_consults = crud.get_consultations(db, skip=0, limit=1000)
    completed = sum(1 for c in all_consults if c.status == "completed")
    pending_actions = sum(
        1 for c in all_consults
        if c.action_items_json and json.loads(c.action_items_json)
    )

    return {
        "total_patients": total_patients,
        "total_consultations": total_consultations,
        "today_consultations": today_consultations,
        "completed_notes": completed,
        "pending_actions": pending_actions,
        "recent_consultations": recent_list,
    }


# ══════════════════════════════════════════════════════════════════════════
# CONSULTATION PROCESSING (existing, updated to persist to DB)
# ══════════════════════════════════════════════════════════════════════════

# ── POST /consult/fast-process (Real-Time Live Transcript → Complete Clinical Package in ~2s) ──

@app.post("/consult/fast-process")
async def fast_process_consultation(
    body: FastProcessRequest,
    db: Session = Depends(get_db),
):
    """
    Ultra-fast single-pass processing for live browser speech or text transcripts.
    Executes in ~2 seconds using Gemini 2.0 Flash:
      - Clinical extraction (chief complaint, symptoms, vitals, Rx)
      - Formatted SOAP Note
      - ICD-10 diagnostic & billing codes
      - Differential Diagnoses (DDx)
      - Allergy & Safety contraindication alerts (cross-referenced with patient history)
      - Patient Discharge Handout (English + Hindi)
      - Action item checklist
    """
    if not body.transcript_text.strip():
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty")

    # Get patient context for clinical cross-checking
    patient_info = None
    if body.patient_id:
        p = crud.get_patient(db, body.patient_id)
        if p:
            patient_info = {
                "id": p.id,
                "name": p.name,
                "age": p.age,
                "gender": p.gender,
                "medical_history": p.medical_history,
                "allergies": p.allergies,
            }

    try:
        # Run single-pass advanced clinical intelligence
        result = run_advanced_clinical_extraction(
            transcript_input=body.transcript_text,
            patient_info=patient_info,
        )

        session_id = uuid.uuid4().hex[:12]
        role_labeled = result.get("role_labeled_transcript", [])
        if not role_labeled:
            role_labeled = [{"speaker": "Consultation", "text": body.transcript_text}]

        # Store session in memory for RAG Q&A
        SESSIONS[session_id] = {
            "audio_path": None,
            "transcript": role_labeled,
            "diarization": [],
            "merged_transcript": role_labeled,
            "extraction_result": {
                "role_mapping": {},
                "role_labeled_transcript": role_labeled,
                "extraction": result.get("extraction", {}),
                "soap_note": result.get("soap_note", ""),
                "icd10_codes": result.get("icd10_codes", []),
                "differential_diagnosis": result.get("differential_diagnosis", []),
                "safety_alerts": result.get("safety_alerts", []),
                "patient_instructions": result.get("patient_instructions", ""),
            },
            "rag_indexed": False,
        }

        # RAG index will be built on-demand if the user asks a follow-up question

        # Persist consultation to DB
        consultation = crud.create_consultation(
            db,
            session_id=session_id,
            audio_filename="live_recording.wav",
            patient_id=body.patient_id,
            transcript=role_labeled,
        )

        crud.update_consultation_extraction(
            db,
            session_id=session_id,
            soap_note=result.get("soap_note", ""),
            extraction={
                **result.get("extraction", {}),
                "icd10_codes": result.get("icd10_codes", []),
                "differential_diagnosis": result.get("differential_diagnosis", []),
                "safety_alerts": result.get("safety_alerts", []),
                "patient_instructions": result.get("patient_instructions", ""),
            },
            action_items=result.get("action_summary", []),
        )

        return {
            "session_id": session_id,
            "consultation_id": consultation.id,
            "role_labeled_transcript": role_labeled,
            "soap_note": result.get("soap_note", ""),
            "extraction": result.get("extraction", {}),
            "icd10_codes": result.get("icd10_codes", []),
            "differential_diagnosis": result.get("differential_diagnosis", []),
            "safety_alerts": result.get("safety_alerts", []),
            "patient_instructions": result.get("patient_instructions", ""),
            "action_summary": result.get("action_summary", []),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clinical processing error: {str(e)}")


@app.post("/consult/upload")
async def upload_consultation(
    file: UploadFile = File(...),
    patient_id: int | None = Query(None),
    model_size: str = Query("base"),
    fast_mode: bool = Query(True),
    db: Session = Depends(get_db),
):
    """
    Upload audio file. When fast_mode=True (default), uses fast Whisper 'base'
    model and runs single-pass clinical analysis without heavy CPU diarization stall.
    """
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        audio_path = tmp.name

        # Fast transcribe
        segments = transcribe(audio_path, model_size=model_size)

        # Diarize (fast fallback if token absent)
        turns = []
        if not fast_mode:
            turns = diarize(audio_path)

        merged = merge_transcript_and_diarization(segments, turns)

        # Create session
        session_id = uuid.uuid4().hex[:12]
        SESSIONS[session_id] = {
            "audio_path": audio_path,
            "transcript": segments,
            "diarization": turns,
            "merged_transcript": merged,
            "extraction_result": None,
            "rag_indexed": False,
        }

        # Patient info
        patient_info = None
        if patient_id:
            p = crud.get_patient(db, patient_id)
            if p:
                patient_info = {
                    "id": p.id,
                    "name": p.name,
                    "age": p.age,
                    "gender": p.gender,
                    "medical_history": p.medical_history,
                    "allergies": p.allergies,
                }

        # Persist consultation to DB
        crud.create_consultation(
            db,
            session_id=session_id,
            audio_filename=file.filename,
            patient_id=patient_id,
            transcript=merged,
        )

        # If fast_mode, automatically generate the full clinical note in single pass!
        clinical_package = None
        if fast_mode:
            try:
                clinical_package = run_advanced_clinical_extraction(merged, patient_info)
                SESSIONS[session_id]["extraction_result"] = clinical_package
                SESSIONS[session_id]["merged_transcript"] = clinical_package.get("role_labeled_transcript", merged)

                crud.update_consultation_extraction(
                    db,
                    session_id=session_id,
                    soap_note=clinical_package.get("soap_note", ""),
                    extraction={
                        **clinical_package.get("extraction", {}),
                        "icd10_codes": clinical_package.get("icd10_codes", []),
                        "differential_diagnosis": clinical_package.get("differential_diagnosis", []),
                        "safety_alerts": clinical_package.get("safety_alerts", []),
                        "patient_instructions": clinical_package.get("patient_instructions", ""),
                    },
                    action_items=clinical_package.get("action_summary", []),
                )
            except Exception as ex_err:
                print(f"[Upload] Fast extraction error (will allow manual retry): {ex_err}")

        response_data = {
            "session_id": session_id,
            "transcript": SESSIONS[session_id]["merged_transcript"],
        }
        if clinical_package:
            response_data.update({
                "soap_note": clinical_package.get("soap_note", ""),
                "extraction": clinical_package.get("extraction", {}),
                "icd10_codes": clinical_package.get("icd10_codes", []),
                "differential_diagnosis": clinical_package.get("differential_diagnosis", []),
                "safety_alerts": clinical_package.get("safety_alerts", []),
                "patient_instructions": clinical_package.get("patient_instructions", ""),
                "action_summary": clinical_package.get("action_summary", []),
            })

        return response_data

    except Exception as e:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /consult/{session_id}/note ───────────────────────────────────────

@app.get("/consult/{session_id}/note")
async def get_clinical_note(session_id: str, db: Session = Depends(get_db)):
    """
    Generate (or return cached) clinical extraction + SOAP note + advanced intelligence
    for a previously uploaded or recorded consultation.
    """
    session = SESSIONS.get(session_id)
    if session and session["extraction_result"] is not None:
        result = session["extraction_result"]
        ext = result.get("extraction", {})
        return {
            "extraction": ext,
            "soap_note": result.get("soap_note", ""),
            "icd10_codes": result.get("icd10_codes", ext.get("icd10_codes", [])),
            "differential_diagnosis": result.get("differential_diagnosis", ext.get("differential_diagnosis", [])),
            "safety_alerts": result.get("safety_alerts", ext.get("safety_alerts", [])),
            "patient_instructions": result.get("patient_instructions", ext.get("patient_instructions", "")),
            "action_summary": result.get("action_summary", ext.get("action_summary", [])),
        }

    # If not in memory, check DB
    c_record = crud.get_consultation_by_session(db, session_id)
    if c_record and c_record.soap_note and c_record.extraction_json:
        try:
            ext = json.loads(c_record.extraction_json)
            actions = json.loads(c_record.action_items_json) if c_record.action_items_json else []
            return {
                "extraction": ext,
                "soap_note": c_record.soap_note,
                "icd10_codes": ext.get("icd10_codes", []),
                "differential_diagnosis": ext.get("differential_diagnosis", []),
                "safety_alerts": ext.get("safety_alerts", []),
                "patient_instructions": ext.get("patient_instructions", ""),
                "action_summary": actions,
            }
        except Exception:
            pass

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Run single-pass advanced extraction
    try:
        result = run_advanced_clinical_extraction(session["merged_transcript"])
        session["extraction_result"] = result
        session["merged_transcript"] = result.get("role_labeled_transcript", session["merged_transcript"])

        crud.update_consultation_extraction(
            db,
            session_id=session_id,
            soap_note=result.get("soap_note", ""),
            extraction={
                **result.get("extraction", {}),
                "icd10_codes": result.get("icd10_codes", []),
                "differential_diagnosis": result.get("differential_diagnosis", []),
                "safety_alerts": result.get("safety_alerts", []),
                "patient_instructions": result.get("patient_instructions", ""),
            },
            action_items=result.get("action_summary", []),
        )

        return {
            "extraction": result.get("extraction", {}),
            "soap_note": result.get("soap_note", ""),
            "icd10_codes": result.get("icd10_codes", []),
            "differential_diagnosis": result.get("differential_diagnosis", []),
            "safety_alerts": result.get("safety_alerts", []),
            "patient_instructions": result.get("patient_instructions", ""),
            "action_summary": result.get("action_summary", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /consult/{session_id}/actions ────────────────────────────────────

@app.get("/consult/{session_id}/actions")
async def get_action_items(session_id: str, db: Session = Depends(get_db)):
    """
    Return action items from the clinical extraction.
    Triggers extraction if not already cached.

    Returns: { actions: [...] }
    """
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Compute note first if not cached
    if session["extraction_result"] is None:
        try:
            result = run_full_extraction(session["merged_transcript"])
            session["extraction_result"] = result
            session["merged_transcript"] = result["role_labeled_transcript"]

            # Persist to DB
            crud.update_consultation_extraction(
                db,
                session_id=session_id,
                soap_note=result["soap_note"],
                extraction=result["extraction"],
                action_items=result["extraction"].get("action_summary", []),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    extraction = session["extraction_result"]["extraction"]
    return {
        "actions": extraction.get("action_summary", []),
    }


# ── POST /consult/{session_id}/ask ──────────────────────────────────────

@app.post("/consult/{session_id}/ask")
async def ask_question_endpoint(session_id: str, body: AskRequest):
    """
    Answer a follow-up question about the consultation using RAG
    (retrieval over transcript + Gemini).

    Body: { "question": "..." }
    Returns: { "answer": "..." }
    """
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build RAG index on first call
    if not session["rag_indexed"]:
        build_index(session["merged_transcript"], session_id)
        session["rag_indexed"] = True

    answer = answer_question(session_id, body.question)
    return AskResponse(answer=answer)


# ══════════════════════════════════════════════════════════════════════════
# REAL-TIME STREAMING (WEBSOCKET)
# ══════════════════════════════════════════════════════════════════════════
#
# Protocol
# --------
# Connect:  ws://host/ws/consult?patient_id=<int>&sample_rate=<int>
#
# Client → server
#   binary frame            Raw little-endian 16-bit mono PCM. Sent
#                           continuously, ~100 ms per frame.
#   {"type":"speaker", "speaker":"Doctor"|"Patient"|"auto"}
#   {"type":"patient", "patient_id":123}
#   {"type":"ask", "question":"...", "request_id":"..."}
#   {"type":"config", "sample_rate":48000}
#   {"type":"finalize"}     Ends the encounter and returns the full package.
#   {"type":"ping"}
#
# Server → client (all JSON, all carrying `type` and `session_id`)
#   session_started · level · speech_state · partial_transcript ·
#   final_utterance · status · clinical_update · safety_alert · answer ·
#   finalized · warning · error · pong
#
# Binary in, JSON out: audio needs zero framing overhead, while events need
# to be self-describing for the browser.


def _resolve_patient_info(db: Session, patient_id: int | None) -> dict | None:
    """Load the patient context used for live allergy/contraindication checks."""
    if not patient_id:
        return None
    p = crud.get_patient(db, patient_id)
    if not p:
        return None
    return {
        "id": p.id,
        "name": p.name,
        "age": p.age,
        "gender": p.gender,
        "medical_history": p.medical_history,
        "allergies": p.allergies,
    }


def _load_patient_info(patient_id: int | None) -> dict | None:
    """Same as `_resolve_patient_info` but managing its own short-lived session."""
    if not patient_id:
        return None
    db = SessionLocal()
    try:
        return _resolve_patient_info(db, patient_id)
    finally:
        db.close()


async def _ws_sender(websocket: WebSocket, session: RealtimeSession) -> None:
    """Drain the session's event queue onto the socket."""
    while True:
        event = await session.events.get()
        await websocket.send_text(json.dumps(event, default=str))


async def _ws_receiver(websocket: WebSocket, session: RealtimeSession, state: dict) -> None:
    """
    Read audio frames and control messages until the client disconnects
    or asks to finalize.
    """
    while True:
        message = await websocket.receive()

        msg_type = message.get("type")
        if msg_type == "websocket.disconnect":
            return

        # ── Audio path ───────────────────────────────────────────────────
        raw = message.get("bytes")
        if raw:
            await session.feed_audio(raw, sample_rate=state["sample_rate"])
            continue

        # ── Control path ─────────────────────────────────────────────────
        text = message.get("text")
        if not text:
            continue

        try:
            command = json.loads(text)
        except json.JSONDecodeError:
            await session.events.put({
                "type": "error",
                "session_id": session.session_id,
                "message": "Control messages must be JSON.",
            })
            continue

        action = command.get("type")

        if action == "config":
            rate = int(command.get("sample_rate") or state["sample_rate"])
            # Guard against a bogus rate turning all audio into noise.
            if 8_000 <= rate <= 192_000:
                state["sample_rate"] = rate
            await session.events.put({
                "type": "status",
                "session_id": session.session_id,
                "stage": "configured",
                "message": f"Ingesting at {state['sample_rate']} Hz.",
            })

        elif action == "speaker":
            session.set_speaker_hint(command.get("speaker"))

        elif action == "patient":
            patient_id = command.get("patient_id")
            patient_id = int(patient_id) if patient_id else None
            info = await asyncio.to_thread(_load_patient_info, patient_id)
            await session.attach_patient(patient_id, info)
            await session.events.put({
                "type": "status",
                "session_id": session.session_id,
                "stage": "patient_linked",
                "message": f"Linked to {info['name']}" if info else "Anonymous encounter.",
            })

        elif action == "ask":
            question = (command.get("question") or "").strip()
            request_id = command.get("request_id")
            if question:
                # Answered off to the side so a slow retrieval never stalls
                # the audio stream.
                asyncio.create_task(_answer_live(session, question, request_id))

        elif action == "finalize":
            await session.finalize()
            event_bus.publish("consultation_finalized", {
                "session_id": session.session_id,
                "patient_id": session.patient_id,
            })
            return

        elif action == "ping":
            await session.events.put({
                "type": "pong",
                "session_id": session.session_id,
                "metrics": session.metrics.snapshot(),
            })


async def _answer_live(session: RealtimeSession, question: str, request_id) -> None:
    """Run a live copilot question and push the answer back."""
    try:
        answer = await session.ask(question)
    except Exception as e:
        answer = f"Could not answer right now: {e}"
    await session.events.put({
        "type": "answer",
        "session_id": session.session_id,
        "request_id": request_id,
        "question": question,
        "answer": answer,
    })


def _load_resume_transcript(session_id: str | None) -> list[dict]:
    """
    Recover a previously streamed transcript so a reconnect can continue it.

    Only turns already committed to the database are returned; anything the
    old connection had buffered but not yet autosaved is genuinely lost.
    """
    if not session_id:
        return []
    db = SessionLocal()
    try:
        record = crud.get_consultation_by_session(db, session_id)
        if not record or not record.transcript_json:
            return []
        turns = json.loads(record.transcript_json)
        return turns if isinstance(turns, list) else []
    except Exception:
        return []
    finally:
        db.close()


@app.websocket("/ws/consult")
async def ws_consult(
    websocket: WebSocket,
    patient_id: int | None = Query(None),
    sample_rate: int = Query(16000),
    session_id: str | None = Query(None),
):
    """
    Live ambient scribe stream.

    Receives microphone PCM and pushes back interim text, committed
    utterances with speaker labels, and a clinical package that is refreshed
    throughout the encounter.

    Passing `session_id` resumes an encounter whose connection dropped: the
    already-saved transcript is loaded back in so the final note covers the
    whole visit instead of only the part after the reconnect.
    """
    await websocket.accept()

    patient_info = await asyncio.to_thread(_load_patient_info, patient_id)
    resume_transcript = await asyncio.to_thread(_load_resume_transcript, session_id)

    # An id is only honoured when there is something to resume, so a client
    # cannot hijack or collide with an unrelated live session.
    resume_id = session_id if resume_transcript else None
    if session_id and registry.get(session_id) is not None:
        resume_id = None

    session = await registry.create(
        session_id=resume_id,
        patient_id=patient_id,
        patient_info=patient_info,
        initial_transcript=resume_transcript if resume_id else None,
    )

    # Mirror the live session into the legacy in-memory store so the existing
    # REST endpoints (/consult/{id}/note, /ask) keep working unchanged.
    SESSIONS[session.session_id] = {
        "audio_path": None,
        "transcript": session.transcript,
        "diarization": [],
        "merged_transcript": session.transcript,
        "extraction_result": None,
        "rag_indexed": True,
        "realtime": True,
    }

    event_bus.publish("consultation_started", {
        "session_id": session.session_id,
        "patient_id": patient_id,
        "patient_name": (patient_info or {}).get("name"),
    })

    state = {"sample_rate": sample_rate if 8_000 <= sample_rate <= 192_000 else 16_000}

    sender = asyncio.create_task(_ws_sender(websocket, session))
    receiver = asyncio.create_task(_ws_receiver(websocket, session, state))

    try:
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            # Surface a genuine crash rather than swallowing it silently.
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                print(f"[ws/consult] {type(exc).__name__}: {exc}")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/consult] Unexpected error: {e}")
    finally:
        for task in (sender, receiver):
            if not task.done():
                task.cancel()

        # A dropped connection is still a real encounter: finalize it so the
        # transcript and note are saved rather than discarded.
        if session.final_package is None and session.transcript:
            try:
                await session.finalize()
            except Exception as e:
                print(f"[ws/consult] Recovery finalize failed: {e}")

        cached = SESSIONS.get(session.session_id)
        if cached is not None:
            cached["merged_transcript"] = list(session.transcript)
            cached["transcript"] = list(session.transcript)
            if session.final_package:
                cached["extraction_result"] = session.final_package

        await registry.remove(session.session_id)
        event_bus.publish("consultation_ended", {"session_id": session.session_id})

        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    """
    Live dashboard feed.

    Pushes a stats snapshot on connect, then streams server-wide events
    (consultations starting, finishing, patient records changing) so open
    dashboards stay current without polling.
    """
    await websocket.accept()
    queue = event_bus.subscribe()

    try:
        snapshot = await asyncio.to_thread(_dashboard_snapshot)
        await websocket.send_text(json.dumps({
            "type": "snapshot",
            **snapshot,
        }, default=str))

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=25.0)
            except asyncio.TimeoutError:
                # Idle keepalive: proxies drop silent WebSockets.
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                continue

            payload = dict(event)
            if payload.get("type") in {
                "consultation_started", "consultation_finalized",
                "consultation_ended", "patient_changed",
            }:
                payload["stats"] = await asyncio.to_thread(_dashboard_stats_only)
            payload["active_sessions"] = len(registry.active_ids())
            await websocket.send_text(json.dumps(payload, default=str))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/dashboard] {type(e).__name__}: {e}")
    finally:
        event_bus.unsubscribe(queue)
        try:
            await websocket.close()
        except Exception:
            pass


def _dashboard_stats_only() -> dict:
    """Counters only — cheap enough to recompute on every broadcast."""
    db = SessionLocal()
    try:
        all_consults = crud.get_consultations(db, skip=0, limit=1000)
        return {
            "total_patients": crud.get_patient_count(db),
            "total_consultations": crud.get_consultation_count(db),
            "today_consultations": crud.get_today_consultation_count(db),
            "completed_notes": sum(1 for c in all_consults if c.status == "completed"),
        }
    finally:
        db.close()


def _dashboard_snapshot() -> dict:
    """Counters plus the recent-consultations list, for the initial push."""
    db = SessionLocal()
    try:
        recent = crud.get_consultations(db, skip=0, limit=5)
        all_consults = crud.get_consultations(db, skip=0, limit=1000)
        return {
            "stats": {
                "total_patients": crud.get_patient_count(db),
                "total_consultations": crud.get_consultation_count(db),
                "today_consultations": crud.get_today_consultation_count(db),
                "completed_notes": sum(1 for c in all_consults if c.status == "completed"),
            },
            "recent_consultations": [_consultation_to_response(c) for c in recent],
            "active_sessions": len(registry.active_ids()),
        }
    finally:
        db.close()


@app.get("/realtime/sessions")
async def list_realtime_sessions():
    """Inspect every streaming session currently open on this server."""
    return {
        "active": len(registry.active_ids()),
        "dashboard_subscribers": event_bus.subscriber_count,
        "sessions": registry.status(),
    }


@app.get("/realtime/capabilities")
async def realtime_capabilities():
    """
    Report how the realtime stack is configured.

    Useful for confirming which Whisper tiers are in play and whether the
    live clinical analysis will run through Gemini or the rule-based fallback.
    """
    from backend.extract import gemini_model_name, is_gemini_available
    from backend.stt import default_final_model, default_partial_model

    defaults = RealtimeConfig()
    return {
        "streaming": True,
        "input_format": "pcm_s16le",
        "target_sample_rate": 16000,
        "whisper_partial_model": default_partial_model(),
        "whisper_final_model": default_final_model(),
        "cuda": torch.cuda.is_available(),
        "live_analysis_engine": gemini_model_name() if is_gemini_available() else "rule-based-fallback",
        "neural_diarization": bool(os.getenv("HF_TOKEN")),
        "tuning": {
            "partial_interval_ms": defaults.partial_interval_ms,
            "vad_hangover_ms": defaults.vad_hangover_ms,
            "clinical_debounce_ms": defaults.clinical_debounce_ms,
            "clinical_min_interval_ms": defaults.clinical_min_interval_ms,
            "max_utterance_ms": defaults.max_utterance_ms,
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# SETTINGS & API KEYS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/settings/status")
async def get_settings_status():
    """Return whether Gemini and HuggingFace tokens are configured."""
    from backend.extract import gemini_model_name, is_gemini_available
    return {
        "gemini_configured": is_gemini_available(),
        "hf_configured": bool(os.getenv("HF_TOKEN")),
        "gemini_model": gemini_model_name(),
    }


from pydantic import BaseModel as PydanticBase

class ApiKeysPayload(PydanticBase):
    gemini_api_key: str | None = None
    hf_token: str | None = None


@app.post("/settings/api-keys")
async def save_api_keys(payload: ApiKeysPayload):
    """Save API keys dynamically to environment and persist to .env."""
    import backend.extract as extract_mod

    if payload.gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = payload.gemini_api_key.strip()
        extract_mod._configured = False
    if payload.hf_token is not None:
        os.environ["HF_TOKEN"] = payload.hf_token.strip()

    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}\n")
            f.write(f"GEMINI_API_KEY={os.environ.get('GEMINI_API_KEY', '')}\n")
            # Preserve tuning overrides. This endpoint rewrites .env wholesale,
            # so anything not echoed back here is silently deleted.
            for key in ("GEMINI_MODEL", "WHISPER_FINAL_MODEL",
                        "WHISPER_PARTIAL_MODEL", "PRELOAD_MODELS"):
                value = os.environ.get(key)
                if value:
                    f.write(f"{key}={value}\n")
    except Exception as e:
        print(f"[Settings] .env write warning: {e}")

    from backend.extract import gemini_model_name

    return {
        "status": "ok",
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "hf_configured": bool(os.environ.get("HF_TOKEN")),
        "gemini_model": gemini_model_name(),
    }


# ══════════════════════════════════════════════════════════════════════════
# HEALTH CHECK & ROOT
# ══════════════════════════════════════════════════════════════════════════

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
_INDEX_PATH = os.path.join(_FRONTEND_DIR, "index.html")

@app.get("/")
async def serve_index():
    if os.path.exists(_INDEX_PATH):
        return FileResponse(_INDEX_PATH)
    return {"status": "ok", "service": "CareNote AI Clinical Scribe"}

# Serve welcome audio for the splash screen
_WELCOME_PATH = os.path.join(_FRONTEND_DIR, "welcome.mp3")

@app.get("/welcome.mp3")
async def serve_welcome_audio():
    if os.path.exists(_WELCOME_PATH):
        return FileResponse(_WELCOME_PATH, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Welcome audio not found")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "CareNote AI Clinical Scribe"}
