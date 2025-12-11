import faiss
import numpy as np
from PyPDF2 import PdfReader
from openai import OpenAI

client = OpenAI()

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
            text += page.extract_text() or ""

        # You can also chunk text (recommended)
        self.text_chunks = [text]

        # Generate embeddings
        vectors = client.embeddings.create(
            model="text-embedding-3-small",
            input=self.text_chunks
        )

        self.embeddings = [v.embedding for v in vectors.data]

        dim = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings).astype("float32"))

    def ask(self, query):
        if self.index is None:
            return "Please upload a PDF first."

        # Embed query
        query_vec = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        # Search FAISS
        D, I = self.index.search(
            np.array([query_vec]).astype("float32"), 
            k=1
        )

        context = self.text_chunks[I[0][0]]

        # Ask the model
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Answer using the provided context only."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )

        return response.choices[0].message.content
