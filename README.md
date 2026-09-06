# 🏥 CareNote AI — Ambient AI Clinical Scribe

An end-to-end prototype that takes a pre-recorded doctor-patient consultation audio file (English / Hindi / Hinglish code-switched) and produces:

- **Speaker-labeled transcript** with Doctor/Patient roles identified
- **Structured clinical extraction** (symptoms, medications, history, vitals, etc.)
- **SOAP clinical note** ready for EHR documentation
- **Action items checklist** for the physician
- **RAG-powered Q&A** — ask follow-up questions about the consultation

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.11+**
- A **HuggingFace account** with accepted terms for:
  - [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
- A **Google Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Setup

```bash
# Clone and navigate to the project directory
git clone https://github.com/shubhamnagare83/CareNote_AI.git
cd CareNote_AI

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
HF_TOKEN=hf_your_huggingface_token_here
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Important:** You must accept the pyannote model terms on HuggingFace before the diarization pipeline will work. Visit the model pages linked above and click "Agree and access repository".

### 4. Run

```bash
# Start the FastAPI server from the project root
uvicorn backend.main:app --reload
```

Then open `frontend/index.html` in your browser (just double-click or use a local file server).

> **Tip:** If you see CORS issues, make sure the backend is running on `http://127.0.0.1:8000`.

### 5. Demo

1. Drop a consultation audio file (WAV, MP3, M4A) into the upload area
2. Wait for transcription + diarization (30s–3min depending on audio length and hardware)
3. View the speaker-labeled transcript (Doctor in blue, Patient in green)
4. Click **"Generate Note"** to produce the SOAP clinical note
5. Check **"Action Items"** for the physician task checklist
6. Use **"Ask AI"** to query anything about the consultation

---

## 🏗️ Architecture & Model Choices

### Speech-to-Text: `faster-whisper`
- **Why:** CTranslate2-optimized Whisper inference is 4× faster than original Whisper with comparable accuracy
- **Code-switching support:** No `language` parameter is forced — the model auto-detects, naturally handling English/Hindi/Hinglish mixed speech
- **Hardware adaptation:** Automatically uses `large-v3` + float16 on GPU, or `medium` + int8 on CPU

### Speaker Diarization: `pyannote.audio`
- **Why:** State-of-the-art neural diarization pipeline with excellent accuracy on conversational audio
- **Model:** `pyannote/speaker-diarization-3.1` — handles overlapping speech and produces clean speaker turns
- **Merge strategy:** Each transcript segment is assigned the speaker with maximum time overlap

### Clinical Extraction & SOAP: Google Gemini (`gemini-2.0-flash`)
- **Why:** Fast, cost-effective, and excellent at structured JSON extraction from multilingual medical text
- **JSON mode:** All structured outputs use `response_mime_type: application/json` for reliable parsing
- **Safety:** Prompts explicitly instruct the model to never fabricate diagnoses or symptoms not present in the transcript
- **Validation:** All JSON responses are validated with Pydantic schemas; failed parses trigger one automatic retry

### RAG Q&A: `sentence-transformers` + `chromadb`
- **Why:** Fully local embeddings (no API cost), fast retrieval, zero infrastructure
- **Model:** `all-MiniLM-L6-v2` — lightweight 80MB model, excellent for semantic similarity
- **Storage:** ChromaDB in-memory client — no persistence needed for single-session demo
- **Strategy:** Transcript is chunked by speaker turn, top-4 relevant turns are retrieved and sent to Gemini for answer generation

---

## 📁 Project Structure

```
clinical-scribe/
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI app with all endpoints
│   ├── stt.py           # Transcription via faster-whisper
│   ├── diarize.py       # Speaker diarization + transcript merge
│   ├── extract.py       # Role resolution + clinical extraction + SOAP
│   ├── rag.py           # RAG index building + Q&A
│   └── schemas.py       # Pydantic validation models
├── frontend/
│   └── index.html       # Single-file frontend (Tailwind + vanilla JS)
├── sample_audio/        # Drop test audio clips here
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚠️ Disclaimers

- **This is a prototype / hackathon demo.** Not validated for clinical use.
- **No data persistence.** All session data lives in memory and is lost on server restart.
- **No authentication.** CORS is wide open for local development.
- **Clinical accuracy depends on transcript quality.** Background noise, heavy accents, or low-quality audio will degrade results.
- The SOAP note explicitly states "Not explicitly documented; clinical correlation advised" when the doctor's assessment isn't clearly stated in the transcript — this is by design to prevent AI-fabricated diagnoses.

---

## 📝 License

This project is provided as-is for demonstration and educational purposes.
