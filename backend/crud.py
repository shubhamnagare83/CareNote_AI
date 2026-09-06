"""
crud.py — CRUD operations for Patient and Consultation models.

All functions accept a SQLAlchemy Session and return ORM objects or raise
appropriate exceptions. Designed to be called from FastAPI endpoint handlers.
"""

import json
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.models import Patient, Consultation
from backend.schemas import PatientCreate, PatientUpdate


# ── Patient CRUD ─────────────────────────────────────────────────────────

def create_patient(db: Session, data: PatientCreate) -> Patient:
    """Create a new patient record."""
    patient = Patient(**data.model_dump(exclude_unset=True))
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def get_patients(db: Session, skip: int = 0, limit: int = 100) -> list[Patient]:
    """Get all patients with optional pagination."""
    return db.query(Patient).order_by(Patient.created_at.desc()).offset(skip).limit(limit).all()


def get_patient(db: Session, patient_id: int) -> Patient | None:
    """Get a single patient by ID."""
    return db.query(Patient).filter(Patient.id == patient_id).first()


def update_patient(db: Session, patient_id: int, data: PatientUpdate) -> Patient | None:
    """Update an existing patient. Returns None if not found."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        return None

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(patient, key, value)

    db.commit()
    db.refresh(patient)
    return patient


def delete_patient(db: Session, patient_id: int) -> bool:
    """Delete a patient. Returns True if deleted, False if not found."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        return False

    db.delete(patient)
    db.commit()
    return True


def search_patients(db: Session, query: str) -> list[Patient]:
    """Search patients by name (case-insensitive partial match)."""
    return (
        db.query(Patient)
        .filter(Patient.name.ilike(f"%{query}%"))
        .order_by(Patient.name)
        .all()
    )


def get_patient_count(db: Session) -> int:
    """Get total number of patients."""
    return db.query(func.count(Patient.id)).scalar() or 0


# ── Consultation CRUD ────────────────────────────────────────────────────

def create_consultation(
    db: Session,
    session_id: str,
    audio_filename: str | None = None,
    patient_id: int | None = None,
    transcript: list[dict] | None = None,
) -> Consultation:
    """Create a new consultation record."""
    consultation = Consultation(
        session_id=session_id,
        patient_id=patient_id,
        audio_filename=audio_filename,
        transcript_json=json.dumps(transcript) if transcript else None,
        status="transcribed",
    )
    db.add(consultation)
    db.commit()
    db.refresh(consultation)
    return consultation


def get_consultations(db: Session, skip: int = 0, limit: int = 100) -> list[Consultation]:
    """Get all consultations with optional pagination."""
    return (
        db.query(Consultation)
        .order_by(Consultation.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_consultation(db: Session, consultation_id: int) -> Consultation | None:
    """Get a single consultation by ID."""
    return db.query(Consultation).filter(Consultation.id == consultation_id).first()


def get_consultation_by_session(db: Session, session_id: str) -> Consultation | None:
    """Get a consultation by session_id."""
    return db.query(Consultation).filter(Consultation.session_id == session_id).first()


def get_consultations_by_patient(db: Session, patient_id: int) -> list[Consultation]:
    """Get all consultations for a specific patient."""
    return (
        db.query(Consultation)
        .filter(Consultation.patient_id == patient_id)
        .order_by(Consultation.created_at.desc())
        .all()
    )


def update_consultation_extraction(
    db: Session,
    session_id: str,
    soap_note: str | None = None,
    extraction: dict | None = None,
    action_items: list | None = None,
) -> Consultation | None:
    """Update a consultation with extraction results."""
    consultation = db.query(Consultation).filter(
        Consultation.session_id == session_id
    ).first()
    if not consultation:
        return None

    if soap_note is not None:
        consultation.soap_note = soap_note
    if extraction is not None:
        consultation.extraction_json = json.dumps(extraction)
    if action_items is not None:
        consultation.action_items_json = json.dumps(action_items)

    consultation.status = "completed" if soap_note else "extracted"
    db.commit()
    db.refresh(consultation)
    return consultation


def delete_consultation(db: Session, consultation_id: int) -> bool:
    """Delete a consultation. Returns True if deleted, False if not found."""
    consultation = db.query(Consultation).filter(
        Consultation.id == consultation_id
    ).first()
    if not consultation:
        return False

    db.delete(consultation)
    db.commit()
    return True


def get_consultation_count(db: Session) -> int:
    """Get total number of consultations."""
    return db.query(func.count(Consultation.id)).scalar() or 0


def get_today_consultation_count(db: Session) -> int:
    """Get number of consultations created today."""
    from datetime import datetime, timezone
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return (
        db.query(func.count(Consultation.id))
        .filter(Consultation.created_at >= today_start)
        .scalar() or 0
    )
