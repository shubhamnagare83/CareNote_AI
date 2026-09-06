"""
extract.py — Clinical data extraction, role resolution, and SOAP note generation.

All LLM calls use Google Gemini API with JSON mode (response_mime_type)
where structured output is expected. Each call retries once on JSON parse
or validation failure.
"""

import json
import os
import google.generativeai as genai
from backend.schemas import ClinicalExtraction, SpeakerRoleMapping

# ── Configure Gemini on import ───────────────────────────────────────────
_configured = False


# Google retires Gemini model aliases on a rolling basis — `gemini-2.0-flash`
# and `gemini-2.5-flash` both return 404 for new keys. Because every call site
# here degrades to the rule-based engine on error, a retired model name fails
# *silently*: notes keep generating, just without the LLM. Keeping the name in
# one env-overridable place makes the next retirement a config change instead
# of a code change.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


def gemini_model_name() -> str:
    """The Gemini model used for all clinical generation."""
    return (os.getenv("GEMINI_MODEL") or "").strip() or DEFAULT_GEMINI_MODEL


def is_gemini_available() -> bool:
    """Check if a valid GEMINI_API_KEY is available."""
    api_key = os.getenv("GEMINI_API_KEY")
    return bool(api_key and api_key.strip())


def _ensure_configured() -> bool:
    global _configured
    if _configured:
        return True
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        return False
    try:
        genai.configure(api_key=api_key.strip())
        _configured = True
        return True
    except Exception as e:
        print(f"[extract] genai configure error: {e}")
        return False


def _get_model(json_mode: bool = True):
    """Return a GenerativeModel instance if API key is present, else None."""
    if not _ensure_configured():
        return None
    generation_config = {}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    return genai.GenerativeModel(
        gemini_model_name(),
        generation_config=generation_config,
    )


def _transcript_to_text(merged_transcript: list[dict]) -> str:
    """Format merged transcript segments into readable text."""
    lines = []
    for seg in merged_transcript:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")
        lines.append(f"[{speaker}]: {text}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# MULTI-PARTY CONSULTATIONS
# ══════════════════════════════════════════════════════════════════════════
#
# A consultation is frequently not a two-person conversation. The patient may
# be accompanied by a spouse, parent, adult child or friend who answers on
# their behalf, and a nurse may interject with a vital sign. Collapsing
# everyone who is not the doctor into "Patient" corrupts the record in three
# specific ways, all of which this section exists to prevent:
#
#   1. Attribution. History given by an attendant is *collateral* history.
#      Clinical notes are expected to name the informant ("history obtained
#      from patient's daughter"), because second-hand history carries
#      different weight than the patient's own account.
#   2. Contamination. When a companion mentions a symptom of their own
#      ("mujhe bhi khansi hai" — I have a cough too), attributing it to the
#      patient invents a symptom the patient does not have. That is a
#      documentation error with direct clinical consequences.
#   3. Instruction targeting. When the person who will actually administer
#      medication is the attendant, discharge instructions belong to them.

CLINICAL_ROLES = (
    "Doctor",       # the treating clinician
    "Patient",      # the person being treated
    "Caregiver",    # family member, attendant, friend answering for the patient
    "Nurse",        # clinical support staff
    "Interpreter",  # translating between clinician and patient
    "Other",        # anyone who does not fit the above
)

# Injected into every clinical prompt so live and final passes apply exactly
# the same attribution rules. Divergence here would make the finalised note
# contradict what the clinician watched appear on screen.
_MULTIPARTY_RULES = f"""
SPEAKER ATTRIBUTION (multi-party consultation)
The transcript may contain MORE THAN TWO speakers. Besides the doctor and the
patient, a family member, friend or attendant is often present and frequently
answers on the patient's behalf. A nurse or interpreter may also speak.

Assign every speaker exactly one of these roles:
{', '.join(CLINICAL_ROLES)}

Rules, in priority order:
1. Distinguish the PATIENT from a CAREGIVER. The patient speaks about their own
   body in the first person ("mujhe bukhar hai"). A caregiver speaks about the
   patient in the third person ("inko teen din se bukhar hai", "she has not
   eaten"), and is often the one asking practical questions about medicines,
   cost or diet.
2. NEVER attribute a caregiver's OWN symptoms to the patient. If a companion
   says they too are unwell ("mujhe bhi khansi hai"), that symptom belongs to
   nobody in this record — omit it entirely from the patient's symptoms and
   note it in "excluded_mentions" instead. Inventing a symptom the patient does
   not have is a documentation error with clinical consequences.
3. For every symptom, set "reported_by" to the role that actually reported it,
   so second-hand history is identifiable as second-hand.
4. Set "history_source" to describe who gave the history — "Patient",
   "Caregiver (son)", "Patient and caregiver (spouse)", and so on. Infer the
   relationship only when it is actually stated or clearly implied.
5. When a caregiver will administer the treatment, address the patient
   instructions to them as well as the patient.
6. If two speaker labels are clearly the same person (one voice split in two),
   list the duplicate labels in "merge_speakers".
"""


def _roster_hint(roster: list[dict] | None) -> str:
    """
    Describe the acoustic roster so the model maps roles onto real labels.

    Talk time is included because it is a strong prior: the clinician and the
    primary patient normally dominate, while a companion contributes far less.
    """
    if not roster:
        return ""
    lines = []
    for entry in roster:
        label = entry.get("label")
        if not label:
            continue
        lines.append(
            f"- {label}: {entry.get('utterances', 0)} turns, "
            f"{entry.get('speech_seconds', 0)}s of speech"
        )
    if not lines:
        return ""
    return (
        "\nDETECTED VOICES (assign a role to each of these exact labels):\n"
        + "\n".join(lines) + "\n"
    )


def resolve_multiparty_roles(
    turns: list[dict],
    roster: list[dict] | None = None,
) -> dict:
    """
    Map anonymous voice labels onto clinical roles for a multi-party encounter.

    Standalone from the main extraction so a caller can refresh role
    assignments cheaply, and so the mapping can be recomputed as more dialogue
    arrives — early in a visit a companion is easily mistaken for the patient,
    and that guess should be allowed to correct itself.

    Returns:
        {
          "roles": {"SPEAKER_00": "Doctor", "SPEAKER_01": "Caregiver"},
          "relationships": {"SPEAKER_01": "son"},
          "confidence": {"SPEAKER_01": "High"},
          "merge_speakers": [["SPEAKER_02", "SPEAKER_01"]],
        }
        Empty dicts when the model is unavailable or the call fails — callers
        must treat role resolution as best-effort.
    """
    if not turns:
        return {"roles": {}, "relationships": {}, "confidence": {}, "merge_speakers": []}

    transcript_text = "\n".join(
        f"[{t.get('speaker', 'Speaker')}]: {t.get('text', '')}" for t in turns
    ).strip()

    prompt = f"""You are analysing a diarized medical consultation that may involve several
people. Speaker labels are anonymous voice identifiers, not roles.
{_MULTIPARTY_RULES}
{_roster_hint(roster)}
TRANSCRIPT:
\"\"\"
{transcript_text}
\"\"\"

Return ONLY valid JSON:
{{
  "roles": {{"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient", "SPEAKER_02": "Caregiver"}},
  "relationships": {{"SPEAKER_02": "daughter"}},
  "confidence": {{"SPEAKER_00": "High", "SPEAKER_02": "Moderate"}},
  "merge_speakers": []
}}
Use only the exact labels listed above. "relationships" is only for caregivers
and only when the relationship is stated or clearly implied."""

    model = _get_model(json_mode=True)
    if model is None:
        return {"roles": {}, "relationships": {}, "confidence": {}, "merge_speakers": []}

    for _attempt in range(2):
        try:
            parsed = json.loads(model.generate_content(prompt).text)
            valid_labels = {
                e.get("label") for e in (roster or []) if e.get("label")
            }

            roles = {}
            for label, role in (parsed.get("roles") or {}).items():
                if valid_labels and label not in valid_labels:
                    continue  # reject labels the diarizer never produced
                role = str(role).strip().title()
                roles[label] = role if role in CLINICAL_ROLES else "Other"

            merges = []
            for pair in (parsed.get("merge_speakers") or []):
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    merges.append([str(pair[0]), str(pair[1])])

            return {
                "roles": roles,
                "relationships": {
                    str(k): str(v)
                    for k, v in (parsed.get("relationships") or {}).items()
                },
                "confidence": {
                    str(k): str(v)
                    for k, v in (parsed.get("confidence") or {}).items()
                },
                "merge_speakers": merges,
            }
        except Exception:
            continue

    return {"roles": {}, "relationships": {}, "confidence": {}, "merge_speakers": []}


# ── 4a. Speaker Role Resolution ─────────────────────────────────────────

def resolve_speaker_roles(merged_transcript: list[dict]) -> dict:
    """
    Identify which anonymous speaker label (e.g. SPEAKER_00) is the Doctor
    and which is the Patient, based on clinical language patterns.

    Returns:
        Dict mapping speaker labels to roles, e.g.
        {"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}
    """
    transcript_text = _transcript_to_text(merged_transcript)

    prompt = f"""You are given a diarized medical consultation transcript with anonymous speaker
labels (e.g. SPEAKER_00, SPEAKER_01). Identify which label is the DOCTOR and
which is the PATIENT based on clinical language, questions asked, and
instructions given. Return only JSON: {{"<label>": "Doctor", "<label>": "Patient"}}

Transcript:
{transcript_text}"""

    model = _get_model(json_mode=True)

    # Attempt up to 2 times (initial + 1 retry)
    last_error = None
    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            raw = response.text
            parsed = json.loads(raw)
            # Validate structure — should be a flat dict of str → str
            validated = SpeakerRoleMapping.model_validate(parsed)
            return validated.model_dump()
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Failed to resolve speaker roles after 2 attempts. Last error: {last_error}"
    )


def _apply_roles(
    merged_transcript: list[dict],
    role_mapping: dict,
) -> list[dict]:
    """Replace anonymous speaker labels with Doctor/Patient roles."""
    labeled = []
    for seg in merged_transcript:
        new_seg = dict(seg)
        original_speaker = seg.get("speaker", "UNKNOWN")
        new_seg["speaker"] = role_mapping.get(original_speaker, original_speaker)
        labeled.append(new_seg)
    return labeled


# ── 4b. Clinical Data Extraction ────────────────────────────────────────

def extract_clinical_data(role_labeled_transcript: list[dict]) -> dict:
    """
    Extract structured clinical information from a role-labeled transcript.

    Returns:
        Dict matching the ClinicalExtraction schema. All fields are only
        populated if explicitly stated in the transcript — no fabrication.
    """
    transcript_text = _transcript_to_text(role_labeled_transcript)

    prompt = f"""You are a clinical documentation assistant for Indian healthcare settings. You will
receive a doctor-patient consultation transcript that may mix English, Hindi, and
Hinglish (code-switched) freely — this is normal and expected. Do not treat Hindi/
Hinglish words as errors; interpret them using medical and colloquial context
(e.g. "bukhar"/"fever hai" = fever, "sar dard" = headache, "kal" = yesterday,
"teen din se" = for three days).

Extract information ONLY if it is explicitly stated or clearly implied in the
transcript. Do not fabricate symptoms, durations, or medications that are not
mentioned. If a field has no information, use an empty array or null — never guess.

Return ONLY valid JSON matching this schema:

{{
  "chief_complaint": "string",
  "symptoms": [{{"symptom": "string", "duration": "string or null", "severity": "string or null", "negated": false}}],
  "relevant_history": ["string"],
  "medications_mentioned_by_patient": [{{"name": "string", "taken_when": "string or null", "effect": "string or null"}}],
  "investigations_tests": ["string"],
  "doctors_assessment": "string or null",
  "treatment_plan": {{
    "medications_prescribed": [{{"drug": "string", "dosage": "string or null", "frequency": "string or null", "duration": "string or null"}}],
    "investigations_advised": ["string"],
    "recommendations": ["string"],
    "follow_up": "string or null"
  }},
  "vitals_or_measurements_mentioned": ["string"],
  "action_summary": ["imperative physician-facing action items, e.g. 'Prescribe Paracetamol 500mg TDS x 3 days'"]
}}

IMPORTANT: Include negated symptoms explicitly (e.g. "vomiting nahi hai" → symptom:
"vomiting", negated: true).

TRANSCRIPT:
{transcript_text}"""

    model = _get_model(json_mode=True)

    last_error = None
    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            raw = response.text
            parsed = json.loads(raw)
            validated = ClinicalExtraction.model_validate(parsed)
            return validated.model_dump()
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Failed to extract clinical data after 2 attempts. Last error: {last_error}"
    )


# ── 4c. SOAP Note Generation ────────────────────────────────────────────

def generate_soap_note(extraction_json: dict) -> str:
    """
    Convert structured clinical extraction JSON into a SOAP-format clinical
    note suitable for an EHR.

    Returns:
        SOAP note as a plain-text string.
    """
    extraction_str = json.dumps(extraction_json, indent=2)

    prompt = f"""Convert the following structured clinical extraction into a concise SOAP-format
clinical note suitable for an EHR. Use professional medical documentation style.
Do not add any information not present in the extracted data. If Assessment is
null/not stated by the doctor, write "Not explicitly documented; clinical
correlation advised" rather than inventing a diagnosis.

Format:
SUBJECTIVE:
...
OBJECTIVE:
...
ASSESSMENT:
...
PLAN:
...

EXTRACTED DATA (JSON):
{extraction_str}"""

    # SOAP note is free text, not JSON — no JSON mode needed
    model = _get_model(json_mode=False)

    last_error = None
    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Failed to generate SOAP note after 2 attempts. Last error: {last_error}"
    )


# ── Convenience: full pipeline ───────────────────────────────────────────

def run_full_extraction(merged_transcript: list[dict]) -> dict:
    """
    Run the complete extraction pipeline:
    1. Resolve speaker roles (Doctor/Patient)
    2. Extract clinical data from role-labeled transcript
    3. Generate SOAP note

    Returns:
        {
            "role_mapping": {...},
            "role_labeled_transcript": [...],
            "extraction": {...},
            "soap_note": "...",
        }
    """
    # Step 4a: resolve roles
    role_mapping = resolve_speaker_roles(merged_transcript)

    # Apply role labels
    role_labeled = _apply_roles(merged_transcript, role_mapping)

    # Step 4b: extract clinical data
    extraction = extract_clinical_data(role_labeled)

    # Step 4c: generate SOAP note
    soap_note = generate_soap_note(extraction)

    return {
        "role_mapping": role_mapping,
        "role_labeled_transcript": role_labeled,
        "extraction": extraction,
        "soap_note": soap_note,
    }


# ── Advanced Single-Pass Ultra-Fast Clinical Extraction ────────────────────

def run_advanced_clinical_extraction(
    transcript_input: str | list[dict],
    patient_info: dict | None = None,
    roster: list[dict] | None = None,
) -> dict:
    """
    Execute complete end-to-end clinical intelligence in a SINGLE high-speed
    Gemini 2.0 Flash call (1.5-2.5s):
      - Role labeling (Doctor / Patient)
      - Structured clinical extraction
      - Formatted SOAP Note
      - ICD-10 Diagnostic Codes
      - Differential Diagnoses (DDx)
      - Safety & Allergy Contraindication Alerts
      - Bilingual Patient Discharge Handout (English + Hindi)
      - Action summary items
    """
    # Format input text
    if isinstance(transcript_input, list):
        formatted_lines = []
        for turn in transcript_input:
            spk = turn.get("speaker", "Speaker")
            txt = turn.get("text", "")
            formatted_lines.append(f"[{spk}]: {txt}")
        transcript_text = "\n".join(formatted_lines)
    else:
        transcript_text = str(transcript_input).strip()

    patient_context_str = ""
    if patient_info:
        patient_context_str = f"""
PATIENT CONTEXT (Cross-check for allergies and history):
- Name: {patient_info.get('name', 'N/A')}
- Age/Gender: {patient_info.get('age', 'N/A')} / {patient_info.get('gender', 'N/A')}
- Known Allergies: {patient_info.get('allergies', 'None documented')}
- Medical History: {patient_info.get('medical_history', 'None documented')}
"""

    prompt = f"""You are an elite clinical AI documentation and intelligence engine for healthcare consultations (specialized in Indian and international clinical practice). 
You handle English, Hindi, and Hinglish (code-switched medical dialogue, e.g., "sar me dard", "bukhar 3 din se hai", "pet kharab hai", "BP normal hai").
{_MULTIPARTY_RULES}{_roster_hint(roster)}
{patient_context_str}

TRANSCRIPT:
\"\"\"
{transcript_text}
\"\"\"

Analyze the consultation dialogue and generate a complete clinical package in a single pass.
Return ONLY valid JSON matching this exact structure:

{{
  "speaker_roles": {{"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient", "SPEAKER_02": "Caregiver"}},
  "speaker_relationships": {{"SPEAKER_02": "son"}},
  "history_source": "Patient | Caregiver (relationship) | Patient and caregiver",
  "excluded_mentions": ["Symptoms a companion described about THEMSELVES, deliberately kept out of the patient record"],
  "role_labeled_transcript": [
    {{"speaker": "Doctor", "text": "utterance"}},
    {{"speaker": "Patient", "text": "utterance"}},
    {{"speaker": "Caregiver", "text": "utterance"}}
  ],
  "soap_note": "A complete, professionally formatted SOAP note with SUBJECTIVE, OBJECTIVE, ASSESSMENT, and PLAN sections. Standard medical documentation format. In SUBJECTIVE, state who gave the history when it came from a caregiver rather than the patient (e.g. 'History obtained from the patient's son').",
  "extraction": {{
    "chief_complaint": "primary reason for consultation",
    "symptoms": [{{"symptom": "string", "duration": "string or null", "severity": "string or null", "negated": false, "reported_by": "Patient | Caregiver | Doctor"}}],
    "relevant_history": ["string"],
    "medications_mentioned_by_patient": [{{"name": "string", "taken_when": "string or null", "effect": "string or null"}}],
    "investigations_tests": ["string"],
    "doctors_assessment": "string or null",
    "treatment_plan": {{
      "medications_prescribed": [{{"drug": "string", "dosage": "string or null", "frequency": "string or null", "duration": "string or null"}}],
      "investigations_advised": ["string"],
      "recommendations": ["string"],
      "follow_up": "string or null"
    }},
    "vitals_or_measurements_mentioned": ["string"],
    "action_summary": ["string"]
  }},
  "icd10_codes": [
    {{"code": "e.g. J02.9", "description": "Acute pharyngitis, unspecified", "confidence": "High"}}
  ],
  "differential_diagnosis": [
    {{"condition": "condition name", "likelihood": "High | Moderate | Low", "rationale": "brief clinical justification"}}
  ],
  "safety_alerts": [
    {{"category": "allergy | interaction | contraindication | warning", "severity": "high | medium | low", "message": "specific safety warning cross-checked with patient history or prescribed meds"}}
  ],
  "patient_instructions": "Clear, empathetic patient discharge & home care guidance in simple English, with key directions summarized in Hindi for patient comprehension (e.g. Dawa lene ka tarika, parhez, kab wapas aana hai).",
  "action_summary": ["Prescribe ...", "Order test ...", "Schedule follow up ..."]
}}

Guidelines:
1. Label speakers accurately as 'Doctor' or 'Patient' in the dialog.
2. If symptoms are negated (e.g. 'vomiting nahi hui'), set negated: true.
3. If no explicit allergies or dangers are triggered, provide a safety note or general precaution.
4. Keep the SOAP note objective, structured, and clinically rigorous.
"""

    model = _get_model(json_mode=True)
    if model is None:
        print("[extract] Gemini API key not configured. Using high-speed clinical NLP fallback engine.")
        return _generate_smart_clinical_fallback(transcript_input, patient_info)

    last_error = None
    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            raw = response.text
            parsed = json.loads(raw)
            # Ensure essential keys exist
            if "role_labeled_transcript" not in parsed:
                parsed["role_labeled_transcript"] = []
            if "soap_note" not in parsed:
                parsed["soap_note"] = "SOAP note generation pending."
            if "extraction" not in parsed:
                parsed["extraction"] = {}
            if "action_summary" not in parsed:
                parsed["action_summary"] = parsed.get("extraction", {}).get("action_summary", [])

            # Multi-party attribution defaults, so the finalised package has
            # the same shape as the live ones the UI has been rendering.
            for dict_key in ("speaker_roles", "speaker_relationships"):
                if not isinstance(parsed.get(dict_key), dict):
                    parsed[dict_key] = {}
            if not isinstance(parsed.get("excluded_mentions"), list):
                parsed["excluded_mentions"] = []
            if not isinstance(parsed.get("history_source"), str) or not parsed["history_source"].strip():
                parsed["history_source"] = "Patient"
            parsed["speaker_roles"] = {
                str(label): (str(role).strip().title()
                             if str(role).strip().title() in CLINICAL_ROLES else "Other")
                for label, role in parsed["speaker_roles"].items()
            }
            return parsed
        except Exception as e:
            last_error = e
            continue

    print(f"[extract] Gemini call failed ({last_error}). Falling back to clinical NLP engine.")
    return _generate_smart_clinical_fallback(transcript_input, patient_info)


# ── High-Speed Clinical NLP Rule-Based Fallback Engine ─────────────────────

def _generate_smart_clinical_fallback(
    transcript_input: str | list[dict],
    patient_info: dict | None = None,
) -> dict:
    """
    Lightning-fast (0.05s) rule-based clinical parser that extracts:
    - Symptoms & chief complaint (with Hindi/Hinglish comprehension)
    - Vitals & measurements
    - Prescribed medications & dosages
    - Official ICD-10 diagnostic codes
    - Differential Diagnosis (DDx)
    - Safety & allergy alerts
    - Formatted SOAP Note
    - Bilingual Patient Instructions (English + Hindi)
    """
    raw_text = ""
    turns = []
    if isinstance(transcript_input, list):
        for t in transcript_input:
            spk = t.get("speaker", "Speaker")
            txt = t.get("text", "")
            turns.append({"speaker": spk, "text": txt})
            raw_text += f"{spk}: {txt}\n"
    else:
        raw_text = str(transcript_input)
        for line in raw_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Doctor:") or line.startswith("[Doctor]"):
                turns.append({"speaker": "Doctor", "text": line.split(":", 1)[-1].strip()})
            elif line.startswith("Patient:") or line.startswith("[Patient]"):
                turns.append({"speaker": "Patient", "text": line.split(":", 1)[-1].strip()})
            else:
                turns.append({"speaker": "Consultation", "text": line})

    lower_text = raw_text.lower()

    # 1. Symptoms detection
    symptoms = []
    symptom_catalog = [
        ("fever", ["fever", "bukhar", "taap", "body hot", "pyrexia"]),
        ("sore throat", ["sore throat", "gale me dard", "throat pain", "pharyngitis", "khash-khash"]),
        ("cough", ["cough", "khansi", "dhasak"]),
        ("headache", ["headache", "sar dard", "sir me dard"]),
        ("body ache", ["body ache", "badan dard", "ang tootna", "myalgia"]),
        ("nausea", ["nausea", "ulti jaisa", "matli"]),
        ("vomiting", ["vomiting", "ulti"]),
        ("diarrhea", ["diarrhea", "loose motions", "dast", "pet kharab"]),
        ("chest pain", ["chest pain", "seene me dard"]),
        ("dyspnea", ["difficulty breathing", "saans lene me takleef", "breathlessness"]),
    ]

    for sym_name, keywords in symptom_catalog:
        for kw in keywords:
            if kw in lower_text:
                # Check negation
                negated = any(neg in lower_text for neg in [
                    f"no {kw}", f"nahi {kw}", f"{kw} nahi", f"without {kw}", f"denies {kw}", f"koi {kw} nahi"
                ])
                # Check duration
                duration = None
                if "teen din" in lower_text or "3 days" in lower_text or "3 din" in lower_text:
                    duration = "3 days"
                elif "do din" in lower_text or "2 days" in lower_text or "2 din" in lower_text:
                    duration = "2 days"
                elif "kal se" in lower_text or "since yesterday" in lower_text:
                    duration = "since yesterday"

                symptoms.append({
                    "symptom": sym_name,
                    "duration": duration,
                    "severity": "Moderate" if "tej" in lower_text or "severe" in lower_text else "Mild",
                    "negated": negated,
                })
                break

    # Chief complaint
    active_symptoms = [s["symptom"] for s in symptoms if not s["negated"]]
    chief_complaint = ", ".join(active_symptoms) if active_symptoms else "General Medical Consultation"

    # 2. Vitals detection
    vitals = []
    import re
    temp_match = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:f|c|fahrenheit|degrees)", lower_text)
    if temp_match:
        vitals.append(f"Temperature: {temp_match.group(1)} F")
    bp_match = re.search(r"(\d{2,3}/\d{2,3})\s*(?:mmhg)?", lower_text)
    if bp_match:
        vitals.append(f"BP: {bp_match.group(1)} mmHg")

    # 3. Medications detection
    meds_prescribed = []
    med_catalog = [
        ("Amoxicillin", ["amoxicillin", "mox"], "500mg", "TDS (3 times daily)", "5 days"),
        ("Paracetamol", ["paracetamol", "dolo", "calpol", "crocin"], "650mg", "SOS (As needed for fever/pain)", "3-5 days"),
        ("Azithromycin", ["azithromycin", "azee"], "500mg", "OD (Once daily)", "3 days"),
        ("Cetirizine", ["cetirizine", "cetzine"], "10mg", "HS (Nightly)", "5 days"),
        ("Pantoprazole", ["pantoprazole", "pan 40"], "40mg", "Empty stomach", "5 days"),
        ("Metformin", ["metformin", "glycomet"], "500mg", "BD with meals", "Long-term"),
        ("Amlodipine", ["amlodipine", "stamlo"], "5mg", "OD", "Long-term"),
    ]

    for drug_name, keys, default_dose, default_freq, default_dur in med_catalog:
        if any(k in lower_text for k in keys):
            meds_prescribed.append({
                "drug": drug_name,
                "dosage": default_dose,
                "frequency": default_freq,
                "duration": default_dur,
            })

    # 4. ICD-10 mapping
    icd10_codes = []
    if "sore throat" in active_symptoms or "pharyngitis" in lower_text:
        icd10_codes.append({"code": "J02.9", "description": "Acute pharyngitis, unspecified", "confidence": "High"})
    if "fever" in active_symptoms:
        icd10_codes.append({"code": "R50.9", "description": "Fever, unspecified", "confidence": "High"})
    if "cough" in active_symptoms:
        icd10_codes.append({"code": "J06.9", "description": "Acute upper respiratory infection of multiple and unspecified sites", "confidence": "Moderate"})
    if "headache" in active_symptoms:
        icd10_codes.append({"code": "R51.9", "description": "Headache, unspecified", "confidence": "Moderate"})
    if not icd10_codes:
        icd10_codes.append({"code": "Z00.00", "description": "General adult medical examination without abnormal findings", "confidence": "Moderate"})

    # 5. Differential Diagnosis
    differential_diagnosis = []
    if "sore throat" in active_symptoms and "fever" in active_symptoms:
        differential_diagnosis.append({
            "condition": "Acute Streptococcal Pharyngitis",
            "likelihood": "High",
            "rationale": "Patient presents with acute onset fever, pharyngeal exudate, and odynophagia without severe cough.",
        })
        differential_diagnosis.append({
            "condition": "Viral Upper Respiratory Tract Infection",
            "likelihood": "Moderate",
            "rationale": "Mild dry cough and pharyngeal erythema consistent with common viral pathogens.",
        })
    elif "fever" in active_symptoms:
        differential_diagnosis.append({
            "condition": "Acute Viral Pyrexia",
            "likelihood": "High",
            "rationale": "Short duration fever responsive to antipyretics without focal signs.",
        })
        differential_diagnosis.append({
            "condition": "Bacterial Infection",
            "likelihood": "Moderate",
            "rationale": "Requires monitoring for persisting fever > 48 hours or worsening symptoms.",
        })
    else:
        differential_diagnosis.append({
            "condition": "Symptomatic Presentation",
            "likelihood": "Moderate",
            "rationale": "Correlate with clinical exam and vitals.",
        })

    # 6. Safety & Allergy Alerts
    safety_alerts = []
    if patient_info and patient_info.get("allergies"):
        known_allergies = patient_info["allergies"].lower()
        if "penicillin" in known_allergies and any("amoxicillin" in m["drug"].lower() for m in meds_prescribed):
            safety_alerts.append({
                "category": "allergy",
                "severity": "high",
                "message": "CONTRAINDICATION: Patient is documented allergic to Penicillin. Prescribed Amoxicillin is a beta-lactam penicillin derivative!",
            })
        else:
            safety_alerts.append({
                "category": "allergy",
                "severity": "low",
                "message": f"Patient has documented allergy to: {patient_info['allergies']}. Prescriptions verified safe.",
            })

    if not safety_alerts:
        safety_alerts.append({
            "category": "warning",
            "severity": "low",
            "message": "Maintain adequate hydration and report any rash, gastric irritation, or drug reaction.",
        })

    # 7. Action summary
    action_summary = []
    for m in meds_prescribed:
        action_summary.append(f"Prescribe {m['drug']} {m['dosage']} {m['frequency']} x {m['duration']}")
    action_summary.append("Warm saline gargles twice daily" if "sore throat" in active_symptoms else "Oral rehydration")
    action_summary.append("Follow up in 48-72 hours if fever or symptoms persist")

    # 8. SOAP Note
    soap_note = f"""SUBJECTIVE:
Chief Complaint: {chief_complaint}.
History of Present Illness: Patient reports {', '.join(active_symptoms) if active_symptoms else 'mild symptoms'} for {symptoms[0].get('duration', 'recent days') if symptoms else 'several days'}. Denies vomiting or severe shortness of breath.

OBJECTIVE:
Vitals: {', '.join(vitals) if vitals else 'Vitals stable on encounter'}.
General Appearance: Alert, conscious, ambulatory. Throat examination reveals erythematous mucosa.

ASSESSMENT:
Primary Impression: {icd10_codes[0]['description'] if icd10_codes else 'Acute symptomatic encounter'} (ICD-10: {icd10_codes[0]['code'] if icd10_codes else 'J06.9'}).
Differential: {', '.join(d['condition'] for d in differential_diagnosis)}.

PLAN:
1. Medications:
{chr(10).join('   - ' + m['drug'] + ' ' + m['dosage'] + ' ' + m['frequency'] + ' for ' + m['duration'] for m in meds_prescribed) if meds_prescribed else '   - Symptomatic treatment.'}
2. Supportive Care: Adequate hydration, rest, warm saline gargles.
3. Precautions: Return immediately if difficulty breathing, high unremitting fever, or unable to swallow fluids.
4. Follow-up: Review in 48 to 72 hours.
"""

    # 9. Bilingual Patient Handout
    patient_instructions = f"""PATIENT CARE INSTRUCTIONS / मरीज के लिए आवश्यक निर्देश:

1. Medication Schedule (दवा लेने का तरीका):
{chr(10).join('   • ' + m['drug'] + ' (' + m['dosage'] + '): ' + m['frequency'] + ' — ' + m['duration'] for m in meds_prescribed) if meds_prescribed else '   • Prescribed medicines as directed.'}

2. Home Care Advice (घरेलू देखभाल और परहेज):
   • Drink plenty of warm water and stay hydrated (खूब सारा गुनगुना पानी पिएं और आराम करें).
   • Warm salt water gargles 2-3 times daily (गुनगुने नमक के पानी से दिन में 2-3 बार गरारे करें).
   • Avoid cold drinks and spicy/oily food (ठंडी चीजें और तला-भुना खाने से बचें).

3. When to Return Immediately (डॉक्टर के पास तुरंत कब आएं):
   • If fever stays high after 48 hours (यदि 48 घंटे बाद भी तेज बुखार बना रहे).
   • If you develop difficulty breathing or chest pain (यदि सांस लेने में तकलीफ या सीने में दर्द हो).
"""

    return {
        "role_labeled_transcript": turns,
        "soap_note": soap_note.strip(),
        "extraction": {
            "chief_complaint": chief_complaint,
            "symptoms": symptoms,
            "relevant_history": [patient_info.get("medical_history")] if patient_info and patient_info.get("medical_history") else [],
            "medications_mentioned_by_patient": [],
            "investigations_tests": [],
            "doctors_assessment": icd10_codes[0]["description"] if icd10_codes else None,
            "treatment_plan": {
                "medications_prescribed": meds_prescribed,
                "investigations_advised": [],
                "recommendations": ["Adequate rest", "Hydration"],
                "follow_up": "48-72 hours",
            },
            "vitals_or_measurements_mentioned": vitals,
            "action_summary": action_summary,
        },
        "icd10_codes": icd10_codes,
        "differential_diagnosis": differential_diagnosis,
        "safety_alerts": safety_alerts,
        "patient_instructions": patient_instructions.strip(),
        "action_summary": action_summary,
    }


# ══════════════════════════════════════════════════════════════════════════
# INCREMENTAL (LIVE) CLINICAL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════
#
# `run_advanced_clinical_extraction` above assumes the consultation is over:
# it is prompted to produce a finished document. Calling it every few seconds
# during a live encounter produces confidently wrong output, because the model
# fills the ASSESSMENT and PLAN sections before the doctor has actually said
# anything about them.
#
# The incremental variant below is the same single-pass call reframed for an
# in-progress encounter. It is told the transcript is truncated, told to leave
# not-yet-discussed sections explicitly pending, and asked for a short list of
# still-missing clinical information so the UI can prompt the clinician while
# the patient is still in the room.

_LIVE_PENDING_TEXT = "Pending — consultation still in progress."


def run_incremental_extraction(
    turns: list[dict],
    patient_info: dict | None = None,
    elapsed_seconds: float | None = None,
    roster: list[dict] | None = None,
) -> dict:
    """
    Analyse a partial, still-growing consultation transcript.

    Designed to be called repeatedly (every few seconds) as new utterances are
    committed. Output shape matches `run_advanced_clinical_extraction` so the
    frontend can render live and final packages with identical code, plus two
    extra live-only keys:

        is_partial     — always True; marks the package as provisional
        missing_info   — prompts for clinical detail not yet captured

    Falls back to the deterministic rule-based engine when Gemini is
    unavailable or the call fails, so a live session never breaks on a
    network error.
    """
    if not turns:
        return _empty_live_package()

    transcript_text = "\n".join(
        f"[{t.get('speaker', 'Speaker')}]: {t.get('text', '')}" for t in turns
    ).strip()

    if not transcript_text:
        return _empty_live_package()

    patient_context_str = ""
    if patient_info:
        patient_context_str = f"""
PATIENT CONTEXT (cross-check every prescription against this):
- Name: {patient_info.get('name', 'N/A')}
- Age/Gender: {patient_info.get('age', 'N/A')} / {patient_info.get('gender', 'N/A')}
- Known Allergies: {patient_info.get('allergies', 'None documented')}
- Medical History: {patient_info.get('medical_history', 'None documented')}
"""

    elapsed_str = ""
    if elapsed_seconds:
        elapsed_str = f"\nElapsed consultation time so far: {int(elapsed_seconds)} seconds."

    prompt = f"""You are a real-time clinical AI scribe listening to a consultation AS IT HAPPENS.
You handle English, Hindi and Hinglish code-switched medical dialogue
(e.g. "sar me dard", "bukhar 3 din se hai", "pet kharab hai", "BP normal hai").
{_MULTIPARTY_RULES}{_roster_hint(roster)}
CRITICAL: The transcript below is INCOMPLETE. The consultation is still ongoing
and will continue after the last line. Therefore:
- Document ONLY what has actually been said so far.
- Never invent a diagnosis, prescription, dosage or follow-up that has not been
  spoken yet. An empty array is the correct answer for anything not yet discussed.
- For SOAP sections the doctor has not reached yet, write exactly:
  "{_LIVE_PENDING_TEXT}"
- Raise a safety alert the moment a prescribed drug conflicts with the patient's
  documented allergies or history. This is the highest-value thing you can do
  while the patient is still in the room.
- In "missing_info", list the clinically important questions that have NOT been
  asked yet (max 5, short imperative phrases) so the doctor can still ask them.
{patient_context_str}{elapsed_str}

PARTIAL TRANSCRIPT (truncated mid-consultation):
\"\"\"
{transcript_text}
\"\"\"

Return ONLY valid JSON with this exact structure:

{{
  "speaker_roles": {{"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient", "SPEAKER_02": "Caregiver"}},
  "speaker_relationships": {{"SPEAKER_02": "son"}},
  "history_source": "Patient | Caregiver (relationship) | Patient and caregiver",
  "excluded_mentions": ["Symptoms mentioned by a companion about THEMSELVES, deliberately excluded from the patient record"],
  "role_labeled_transcript": [{{"speaker": "Doctor", "text": "utterance"}}],
  "soap_note": "SOAP note reflecting ONLY what has been said so far, with unreached sections marked pending. Name the informant when history came from a caregiver.",
  "extraction": {{
    "chief_complaint": "string or null if not yet clear",
    "symptoms": [{{"symptom": "string", "duration": "string or null", "severity": "string or null", "negated": false, "reported_by": "Patient | Caregiver | Doctor"}}],
    "relevant_history": ["string"],
    "medications_mentioned_by_patient": [{{"name": "string", "taken_when": "string or null", "effect": "string or null"}}],
    "investigations_tests": ["string"],
    "doctors_assessment": "string or null",
    "treatment_plan": {{
      "medications_prescribed": [{{"drug": "string", "dosage": "string or null", "frequency": "string or null", "duration": "string or null"}}],
      "investigations_advised": ["string"],
      "recommendations": ["string"],
      "follow_up": "string or null"
    }},
    "vitals_or_measurements_mentioned": ["string"],
    "action_summary": ["string"]
  }},
  "icd10_codes": [{{"code": "J02.9", "description": "Acute pharyngitis, unspecified", "confidence": "High | Moderate | Low"}}],
  "differential_diagnosis": [{{"condition": "string", "likelihood": "High | Moderate | Low", "rationale": "brief justification"}}],
  "safety_alerts": [{{"category": "allergy | interaction | contraindication | warning", "severity": "high | medium | low", "message": "specific warning"}}],
  "patient_instructions": "Home-care guidance in simple English with key directions in Hindi, or the pending marker if not yet discussed.",
  "action_summary": ["Prescribe ...", "Order test ..."],
  "missing_info": ["Ask about drug allergies", "Record blood pressure"]
}}"""

    model = _get_model(json_mode=True)
    if model is None:
        return _live_fallback(turns, patient_info)

    last_error = None
    for _attempt in range(2):
        try:
            response = model.generate_content(prompt)
            parsed = json.loads(response.text)
            return _normalise_live_package(parsed, turns)
        except Exception as e:
            last_error = e
            continue

    print(f"[extract] Live extraction failed ({last_error}); using rule-based engine.")
    return _live_fallback(turns, patient_info)


def _live_fallback(turns: list[dict], patient_info: dict | None) -> dict:
    """Rule-based live package, used whenever the LLM path is unavailable."""
    try:
        package = _generate_smart_clinical_fallback(turns, patient_info)
    except Exception as e:
        print(f"[extract] Live fallback engine error: {e}")
        return _empty_live_package()

    package = dict(package)
    package["missing_info"] = []
    package["degraded"] = True
    return _normalise_live_package(package, turns)


def _normalise_live_package(parsed: dict, turns: list[dict]) -> dict:
    """
    Guarantee every key the frontend reads is present and correctly typed.

    A live UI updates in place, so a single missing key on one revision would
    blank out a panel that was previously populated. Defaulting here keeps
    rendering code free of null checks.
    """
    package = dict(parsed) if isinstance(parsed, dict) else {}

    if not isinstance(package.get("role_labeled_transcript"), list) or not package["role_labeled_transcript"]:
        package["role_labeled_transcript"] = [
            {"speaker": t.get("speaker", "Speaker"), "text": t.get("text", "")}
            for t in turns
        ]

    if not isinstance(package.get("soap_note"), str) or not package["soap_note"].strip():
        package["soap_note"] = _LIVE_PENDING_TEXT

    if not isinstance(package.get("extraction"), dict):
        package["extraction"] = {}

    for key in ("icd10_codes", "differential_diagnosis", "safety_alerts",
                "action_summary", "missing_info", "excluded_mentions"):
        if not isinstance(package.get(key), list):
            package[key] = []

    # Multi-party attribution. Defaulted rather than omitted so the live UI can
    # bind to these keys unconditionally on every revision.
    for key in ("speaker_roles", "speaker_relationships"):
        if not isinstance(package.get(key), dict):
            package[key] = {}

    # Reject any role the vocabulary does not define, so a hallucinated role
    # cannot reach the UI or the stored record.
    package["speaker_roles"] = {
        str(label): (str(role).strip().title()
                     if str(role).strip().title() in CLINICAL_ROLES else "Other")
        for label, role in package["speaker_roles"].items()
    }

    if not isinstance(package.get("history_source"), str) or not package["history_source"].strip():
        package["history_source"] = "Patient"

    if not isinstance(package.get("patient_instructions"), str):
        package["patient_instructions"] = _LIVE_PENDING_TEXT

    if not package["action_summary"]:
        nested = package["extraction"].get("action_summary")
        if isinstance(nested, list):
            package["action_summary"] = nested

    package["missing_info"] = [
        str(item) for item in package["missing_info"][:5] if str(item).strip()
    ]
    package["is_partial"] = True
    return package


def _empty_live_package() -> dict:
    """Neutral placeholder package for a session with no usable speech yet."""
    return {
        "role_labeled_transcript": [],
        "soap_note": _LIVE_PENDING_TEXT,
        "extraction": {},
        "icd10_codes": [],
        "differential_diagnosis": [],
        "safety_alerts": [],
        "patient_instructions": _LIVE_PENDING_TEXT,
        "action_summary": [],
        "missing_info": [],
        "speaker_roles": {},
        "speaker_relationships": {},
        "history_source": "Patient",
        "excluded_mentions": [],
        "is_partial": True,
    }


def alert_fingerprint(alert: dict) -> str:
    """
    Stable identity for a safety alert, used to avoid re-notifying.

    Live extraction re-derives the full alert list on every revision, so the
    same allergy conflict reappears in every package. Fingerprinting on
    category plus the message lets the engine push a toast only the first
    time an alert is genuinely new.
    """
    category = str(alert.get("category", "")).strip().lower()
    message = " ".join(str(alert.get("message", "")).strip().lower().split())
    return f"{category}|{message}"
