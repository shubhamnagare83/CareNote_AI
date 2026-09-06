"""
models.py — SQLAlchemy ORM models for CareNote AI.

Models:
  - Patient:       Demographics, medical history, allergies
  - Consultation:  Audio session linked to a patient, stores transcript,
                   extraction, SOAP note, and action items as JSON
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Enum as SAEnum,
)
from sqlalchemy.orm import relationship
from backend.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    phone = Column(String(30), nullable=True)
    email = Column(String(100), nullable=True)
    blood_group = Column(String(10), nullable=True)
    medical_history = Column(Text, nullable=True)
    allergies = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationship
    consultations = relationship(
        "Consultation", back_populates="patient", cascade="all, delete-orphan"
    )


class Consultation(Base):
    __tablename__ = "consultations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(50), unique=True, index=True, nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    audio_filename = Column(String(255), nullable=True)
    transcript_json = Column(Text, nullable=True)      # JSON string of merged transcript
    soap_note = Column(Text, nullable=True)
    extraction_json = Column(Text, nullable=True)       # JSON string of clinical extraction
    action_items_json = Column(Text, nullable=True)     # JSON string of action items list
    status = Column(
        String(30),
        default="transcribed",
        nullable=False,
    )  # transcribed | extracted | completed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship
    patient = relationship("Patient", back_populates="consultations")
