import os
import re
import json
import tempfile
import requests
import fitz
import pytesseract
from PIL import Image
from docx import Document
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from fastapi import FastAPI, Request

app = FastAPI()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Utility function: Download file from URL to temp directory
def download_file(url: str, dest_dir: str) -> str:
    ext = url.split('.')[-1].split('?')[0]
    local_path = os.path.join(dest_dir, f"file_{abs(hash(url))}.{ext}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return local_path

# Extract text from PDF, DOCX, or Images
def extract_text(file_path: str, max_pages: int = 3) -> str:
    ext = file_path.split('.')[-1].lower()
    if ext == "pdf":
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc[:max_pages])
    elif ext == "docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext in {"jpg", "jpeg", "png"}:
        return pytesseract.image_to_string(Image.open(file_path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Extract parameters like age, gender, procedure, location, policy_duration from text
def extract_params(text: str) -> dict:
    age_m = re.search(r"(\d{2})[- ]?year[- ]?old", text, re.IGNORECASE)
    gender_m = re.search(r"\b(male|female)\b", text, re.IGNORECASE)
    proc_m = re.search(r"(\w+(?:\s\w+)*\s(?:surgery|replacement|operation|treatment))", text, re.IGNORECASE)
    loc_m = re.search(r"(?:in|at)\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", text)
    dur_m = re.search(r"(\d+)[- ]?(?:month|year)[- ]?old.*?insurance", text, re.IGNORECASE)
    return {
        "age": int(age_m.group(1)) if age_m else None,
        "gender": gender_m.group(1).lower() if gender_m else None,
        "procedure": proc_m.group(1).strip() if proc_m else None,
        "location": loc_m.group(1).strip() if loc_m else None,
        "policy_duration": (
            dur_m.group(1) + (" months" if "month" in dur_m.group(0) else " years")
        ) if dur_m else None
    }

# Chunk large text into overlapping pieces
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Prepare FAISS index from list of policy document file paths
def prepare_policy_index(policy_file_paths: list) -> tuple:
    all_chunks, chunk_sources = [], []
    for path in policy_file_paths:
        text = extract_text(path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        chunk_sources.extend([os.path.basename(path)] * len(chunks))
    embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return all_chunks, chunk_sources, index

# Semantic search over the FAISS index for a query string
def semantic_search(query: str, chunks: list, chunk_sources: list, index, top_k: int = 3) -> list:
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [(chunks[i], chunk_sources[i]) for i in I[0]]

# Call Gemini LLM for final decision
def get_llm_decision_gemini(structured_json: dict, retrieved_clauses: list, gemini_api_key: str) -> str:
    genai.configure(api_key=gemini_api_key)
    llm = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are an insurance claim decision model.

Claim Info:
{json.dumps(structured_json, indent=2)}

Relevant Policy Clauses:
{retrieved_clauses[0][0]}
{retrieved_clauses[1][0] if len(retrieved_clauses) > 1 else ''}
{retrieved_clauses[2][0] if len(retrieved_clauses) > 2 else ''}

Your task is to:
1. Decide if the claim should be approved or rejected
2. Mention amount if applicable (else null)
3. Give clear justification pointing to the relevant clauses

Respond only in JSON:
{{"Decision": "...", "Amount": "...", "Justification": "..."}}
"""
    response = llm.generate_content(prompt)
    return response.text

# The FastAPI /hackrx/run endpoint
@app.post("/hackrx/run")
async def hackrx_run(request: Request):
    data = await request.json()
    document_urls = data.get("documents")
    questions = data.get("questions", [])

    if not document_urls:
        return {"error": "No documents provided."}

    if isinstance(document_urls, str):
        document_urls = [document_urls]

    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        return {"error": "API key not configured in environment variables."}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download all policy docs
        policy_paths = [download_file(url, tmpdir) for url in document_urls]
        # Extract text and build FAISS index once per request
        chunks, chunk_sources, index = prepare_policy_index(policy_paths)

        answers = []
        for question in questions:
            # Extract structured info from question (optional; can also use raw question text)
            structured_query = extract_params(question)
            # Compose query text for semantic search
            query_text = " ".join([str(v) for v in structured_query.values() if v])
            # Retrieve top relevant clauses
            retrieved_clauses = semantic_search(query_text, chunks, chunk_sources, index)
            # Get final decision from Gemini
            answer = get_llm_decision_gemini(structured_query, retrieved_clauses, gemini_api_key)
            answers.append(answer)

    return {"answers": answers}
