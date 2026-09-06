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

import json
import os
import uuid
import tempfile
import shutil

from dotenv import load_dotenv

# Load .env before any module that reads env vars
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.stt import transcribe
from backend.diarize import diarize, merge_transcript_and_diarization
from backend.extract import run_full_extraction, run_advanced_clinical_extraction
from backend.rag import build_index, answer_question
from backend.schemas import (
    AskRequest, AskResponse,
    PatientCreate, PatientUpdate, PatientResponse, ConsultationResponse,
    FastProcessRequest,
)
from backend.database import get_db, init_db
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
def on_startup():
    """Create all database tables on application startup."""
    init_db()


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
# SETTINGS & API KEYS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/settings/status")
async def get_settings_status():
    """Return whether Gemini and HuggingFace tokens are configured."""
    from backend.extract import is_gemini_available
    return {
        "gemini_configured": is_gemini_available(),
        "hf_configured": bool(os.getenv("HF_TOKEN")),
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
    except Exception as e:
        print(f"[Settings] .env write warning: {e}")

    return {
        "status": "ok",
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "hf_configured": bool(os.environ.get("HF_TOKEN")),
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

@app.get("/health")
async def health():
    return {"status": "ok", "service": "CareNote AI Clinical Scribe"}
