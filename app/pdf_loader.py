# app/pdf_loader.py

import os
from typing import Optional

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Directory to store persisted vectorstores
DEFAULT_INDEX_DIR = "vectorstores"

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def create_vectorstore_from_pdf(
    pdf_path: str,
    index_name: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    embedding_model: Optional[OpenAIEmbeddings] = None,
    persist: bool = True,
) -> FAISS:
    """
    Load a PDF, split it into chunks, embed chunks with OpenAIEmbeddings,
    create a FAISS vectorstore, and optionally persist it.

    Returns a LangChain FAISS vectorstore instance.
    """
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()

    # Where to save the index
    index_dir = os.path.join(DEFAULT_INDEX_DIR, index_name)
    _ensure_dir(DEFAULT_INDEX_DIR)

    # If index already exists on disk, load and return
    try:
        if os.path.isdir(index_dir) and len(os.listdir(index_dir)) > 0:
            vs = FAISS.load_local(index_dir, embedding_model)
            return vs
    except Exception:
        # If loading fails, we'll rebuild below
        pass

    # Load documents from PDF (fallback to TextLoader if needed)
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except Exception as e:
        # Fallback: try to read as plain text
        try:
            loader = TextLoader(pdf_path, encoding="utf8")
            docs = loader.load()
        except Exception as e2:
            raise RuntimeError(f"Failed to load document: {e}; fallback error: {e2}")

    # Split documents into chunks suitable for embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    split_docs = text_splitter.split_documents(docs)

    # Create FAISS vectorstore from documents
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    # Persist to disk for faster startup next time
    if persist:
        try:
            vectorstore.save_local(index_dir)
        except Exception:
            # older langchain/FAISS versions might not support save_local; attempt manual fallback
            try:
                import pickle
                with open(os.path.join(index_dir, "vectorstore.pkl"), "wb") as f:
                    pickle.dump(vectorstore, f)
            except Exception:
                # if persistence fails, continue without failing the whole flow
                pass

    return vectorstore
