"""
schemas.py — Pydantic models for structured clinical data validation.

These models validate the JSON output from Gemini API calls.
If validation fails, the caller retries once before raising.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Clinical Extraction Schemas ──────────────────────────────────────────

class Symptom(BaseModel):
    symptom: str
    duration: Optional[str] = None
    severity: Optional[str] = None
    negated: bool = False


class PatientMedication(BaseModel):
    name: str
    taken_when: Optional[str] = None
    effect: Optional[str] = None


class PrescribedMedication(BaseModel):
    drug: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None


class TreatmentPlan(BaseModel):
    medications_prescribed: list[PrescribedMedication] = Field(default_factory=list)
    investigations_advised: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    follow_up: Optional[str] = None


class ClinicalExtraction(BaseModel):
    """
    Full structured extraction from a doctor-patient consultation.
    Every field is optional or defaults to empty — the model must never
    fabricate data that isn't explicitly present in the transcript.
    """
    chief_complaint: Optional[str] = None
    symptoms: list[Symptom] = Field(default_factory=list)
    relevant_history: list[str] = Field(default_factory=list)
    medications_mentioned_by_patient: list[PatientMedication] = Field(default_factory=list)
    investigations_tests: list[str] = Field(default_factory=list)
    doctors_assessment: Optional[str] = None
    treatment_plan: TreatmentPlan = Field(default_factory=TreatmentPlan)
    vitals_or_measurements_mentioned: list[str] = Field(default_factory=list)
    action_summary: list[str] = Field(default_factory=list)


class SpeakerRoleMapping(BaseModel):
    """
    Validates Gemini's speaker-role resolution output.
    Expects a dict like {"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}.
    We allow extra speakers (e.g. SPEAKER_02 might be a nurse) via model_config.
    """
    model_config = {"extra": "allow"}

    # At least one mapping should exist; validated at call site, not here.


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    question: str


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""
    answer: str


# ── Patient CRUD Schemas ─────────────────────────────────────────────────

class PatientCreate(BaseModel):
    """Schema for creating a new patient."""
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    blood_group: Optional[str] = None
    medical_history: Optional[str] = None
    allergies: Optional[str] = None


class PatientUpdate(BaseModel):
    """Schema for updating an existing patient. All fields optional."""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    blood_group: Optional[str] = None
    medical_history: Optional[str] = None
    allergies: Optional[str] = None


class PatientResponse(BaseModel):
    """Schema for patient response with all fields."""
    id: int
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    blood_group: Optional[str] = None
    medical_history: Optional[str] = None
    allergies: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    consultation_count: int = 0

    model_config = {"from_attributes": True}


class ConsultationResponse(BaseModel):
    """Schema for consultation response."""
    id: int
    session_id: str
    patient_id: Optional[int] = None
    patient_name: Optional[str] = None
    audio_filename: Optional[str] = None
    status: str = "transcribed"
    created_at: Optional[datetime] = None
    has_transcript: bool = False
    has_soap_note: bool = False
    has_extraction: bool = False

    model_config = {"from_attributes": True}


# ── Advanced Clinical Schemas ───────────────────────────────────────────

class ICD10Code(BaseModel):
    code: str
    description: str
    confidence: Optional[str] = "High"


class DifferentialDiagnosis(BaseModel):
    condition: str
    likelihood: str  # "High", "Moderate", "Low"
    rationale: str


class SafetyAlert(BaseModel):
    category: str  # "allergy", "interaction", "contraindication", "warning"
    severity: str  # "high", "medium", "low"
    message: str


class FastProcessRequest(BaseModel):
    """Request schema for ultra-fast single-pass processing of live or text transcript."""
    patient_id: Optional[int] = None
    transcript_text: str
    transcript_segments: Optional[list[dict]] = None


class AdvancedExtractionResult(BaseModel):
    role_labeled_transcript: list[dict] = Field(default_factory=list)
    soap_note: str
    extraction: ClinicalExtraction
    icd10_codes: list[ICD10Code] = Field(default_factory=list)
    differential_diagnosis: list[DifferentialDiagnosis] = Field(default_factory=list)
    safety_alerts: list[SafetyAlert] = Field(default_factory=list)
    patient_instructions: Optional[str] = None
    action_summary: list[str] = Field(default_factory=list)
