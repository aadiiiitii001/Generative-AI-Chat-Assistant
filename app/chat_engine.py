import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def clean_text(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def split_text(text, chunk_size=1000, chunk_overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


class ChatEngine:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.index = None
        self.text_chunks = []
        self.embeddings = []

    def load_pdf(self, pdf_file):
        reader = PdfReader(pdf_file)
        text = ""

        for page in reader.pages:
            try:
                extracted = page.extract_text() or ""
                text += extracted
            except:
                pass

        text = clean_text(text)
        self.text_chunks = split_text(text, chunk_size=800, chunk_overlap=100)

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=self.text_chunks
        )

        self.embeddings = [item.embedding for item in response.data]

        dim = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings).astype("float32"))

    def ask(self, query):
        if self.index is None:
            return "Please upload a PDF first."

        query_vec = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        D, I = self.index.search(np.array([query_vec]).astype("float32"), k=1)
        best_chunk = self.text_chunks[I[0][0]]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Answer ONLY using the provided context."},
                {"role": "user", "content": f"Context:\n{best_chunk}\n\nQuestion: {query}"}
            ]
        )

        return response.choices[0].message.content
