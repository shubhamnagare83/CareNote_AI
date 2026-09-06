"""
rag.py — RAG (Retrieval-Augmented Generation) Q&A over consultation transcripts.

Uses sentence-transformers (all-MiniLM-L6-v2) for local embeddings and
ChromaDB in-memory for vector storage. Answers questions by retrieving the
top-4 most relevant transcript turns and prompting Gemini.
"""

import json
import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from typing import Any

# ── Singletons ───────────────────────────────────────────────────────────
_embedding_model: SentenceTransformer | None = None
_chroma_client: Any = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        # In-memory client — no persistence, single-session demo
        _chroma_client = chromadb.Client()
    return _chroma_client


def _ensure_gemini_configured() -> bool:
    """Make sure Gemini SDK is configured (idempotent). Returns True if configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        return False
    try:
        genai.configure(api_key=api_key.strip())
        return True
    except Exception:
        return False


# ── Build Index ──────────────────────────────────────────────────────────

def build_index(merged_transcript: list[dict], session_id: str) -> None:
    """
    Chunk the transcript by speaker turn, embed each chunk, and upsert
    into a ChromaDB in-memory collection keyed by session_id.

    Args:
        merged_transcript: List of dicts with keys: speaker, start, end, text.
        session_id:        Unique session identifier for the collection name.
    """
    model = _get_embedding_model()
    client = _get_chroma_client()

    # Create or get collection for this session
    # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores
    collection_name = f"session_{session_id[:50]}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # If already populated, skip re-indexing
    if collection.count() > 0:
        return

    # Build documents: one per speaker turn
    documents = []
    ids = []
    metadatas = []

    for i, seg in enumerate(merged_transcript):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")
        doc = f"[{speaker}]: {text}"
        documents.append(doc)
        ids.append(f"turn_{i}")
        metadatas.append({
            "speaker": speaker,
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
        })

    if not documents:
        return

    # Embed all documents
    embeddings = model.encode(documents, show_progress_bar=False).tolist()

    # Upsert into collection
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# ── Answer Question ─────────────────────────────────────────────────────

def answer_question(session_id: str, question: str) -> str:
    """
    Retrieve the top-4 relevant transcript turns for the question and
    ask Gemini to answer using ONLY those excerpts.

    Args:
        session_id: The session whose transcript index to search.
        question:   User's natural-language question.

    Returns:
        Answer string from Gemini, or a "not mentioned" message.
    """
    _ensure_gemini_configured()
    model_emb = _get_embedding_model()
    client = _get_chroma_client()

    collection_name = f"session_{session_id[:50]}"

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        return (
            "No transcript index found for this session. "
            "Please upload and process an audio file first."
        )

    # Embed the question
    q_embedding = model_emb.encode([question], show_progress_bar=False).tolist()

    # Retrieve top-4 relevant turns
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=min(4, collection.count()),
    )

    # Format retrieved chunks
    retrieved_docs = results.get("documents", [[]])[0]
    if not retrieved_docs:
        return "No relevant transcript segments found."

    excerpts = "\n".join(retrieved_docs)

    # If Gemini is not configured, return relevant semantic excerpts directly
    if not _ensure_gemini_configured():
        return f"Excerpts retrieved from transcript:\n{excerpts}\n\n[Add GEMINI_API_KEY in Settings to enable generative conversational synthesis]"

    # Call Gemini for the answer
    prompt = f"""Answer the question using ONLY the following transcript excerpts. If the answer
isn't present in the excerpts, say "Not mentioned in the consultation."

EXCERPTS:
{excerpts}

QUESTION: {question}"""

    gemini_model = genai.GenerativeModel("gemini-2.0-flash")

    last_error = None
    for attempt in range(2):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            last_error = e
            continue

    return f"Excerpts retrieved from transcript:\n{excerpts}"
