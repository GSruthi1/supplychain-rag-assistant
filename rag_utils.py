import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
import requests
from pypdf import PdfReader


OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"
GEN_MODEL = "llama3.2"

PDF_DIR = Path("data/pdfs")
STORAGE_DIR = Path("storage")
INDEX_PATH = STORAGE_DIR / "faiss.index"
META_PATH = STORAGE_DIR / "metadata.json"


def ensure_storage():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\b\w\s){2,}\w\b', lambda m: m.group(0).replace(" ", ""), text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []

    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        if cleaned:
            pages.append(f"[Page {i + 1}]\n{cleaned}")

    return "\n\n".join(pages).strip()


def load_all_pdfs(pdf_dir: Path) -> List[Dict]:
    docs = []
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        text = extract_text_from_pdf(pdf_file)
        if text.strip():
            docs.append({
                "source": pdf_file.name,
                "text": text
            })
    return docs


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_chunks(docs: List[Dict]) -> List[Dict]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": len(all_chunks),
                "source": doc["source"],
                "chunk_index": idx,
                "text": chunk
            })
    return all_chunks


def get_embeddings(texts: List[str], model: str = EMBED_MODEL, batch_size: int = 32) -> np.ndarray:
    all_embeddings = []

    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, total, batch_size), start=1):
        end = min(start + batch_size, total)
        batch = texts[start:end]

        print(f"Embedding batch {batch_num}/{total_batches}...")

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": batch},
            timeout=300
        )
        response.raise_for_status()

        data = response.json()
        all_embeddings.extend(data["embeddings"])

    return np.array(all_embeddings, dtype="float32")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def save_index_and_metadata(index: faiss.Index, metadata: List[Dict]):
    ensure_storage()
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_index_and_metadata() -> Tuple[faiss.Index, List[Dict]]:
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index or metadata not found. Run ingest.py first.")

    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def search_similar_chunks(query: str, top_k: int = 5) -> List[Dict]:
    index, metadata = load_index_and_metadata()
    query_vec = get_embeddings([query])
    query_vec = normalize_vectors(query_vec)

    distances, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def generate_answer(query: str, retrieved_chunks: List[Dict], model: str = GEN_MODEL) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk['source']} | Chunk {chunk['chunk_index']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
You are answering questions using ONLY the provided context.

Rules:
- Give a clear and direct answer
- Do NOT say "according to source 1"
- At the end, list sources cleanly
- If not found, say: "Not found in documents"

Question:
{query}

Context:
{context}

Answer:
""".strip()

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["response"].strip()