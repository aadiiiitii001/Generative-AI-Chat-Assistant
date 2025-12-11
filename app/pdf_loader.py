# app/pdf_loader.py

import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from openai import OpenAI

client = OpenAI()

DEFAULT_INDEX_DIR = "vectorstores"


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# 🔥 FIX: Remove bad Unicode characters that cause UTF-8 errors
def clean_text(text: str) -> str:
    # Removes invalid surrogate characters
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def split_text(text, chunk_size=1000, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


def create_vectorstore_from_pdf(
    pdf_path: str,
    index_name: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    persist: bool = True
):
    _ensure_dir(DEFAULT_INDEX_DIR)
    index_dir = os.path.join(DEFAULT_INDEX_DIR, index_name)

    # If cached FAISS index exists, load it
    if os.path.exists(os.path.join(index_dir, "faiss.index")):
        faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "chunks.txt"), "r", encoding="utf8") as f:
            chunks = f.read().split("<SPLIT>")
        return faiss_index, chunks

    # Load PDF
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
            text += extracted
        except:
            continue

    # 🔥 FIX: Clean the entire text before splitting
    text = clean_text(text)

    # Split into chunks
    chunks = split_text(text, chunk_size, chunk_overlap)

    # Embed chunks
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    vectors = [item.embedding for item in response.data]
    vectors_np = np.array(vectors).astype("float32")

    # Create FAISS index
    dim = vectors_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors_np)

    # Persist
    if persist:
        _ensure_dir(index_dir)
        faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "chunks.txt"), "w", encoding="utf8") as f:
            f.write("<SPLIT>".join(chunks))

    return index, chunks
