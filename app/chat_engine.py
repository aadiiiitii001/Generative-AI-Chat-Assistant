from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from PyPDF2 import PdfReader


class ChatEngine:
    """Chat engine for PDF question answering."""

    def __init__(self, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.vectorstore = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.chain = None

    def load_pdf(self, pdf_file):
        """Extract text, create embeddings and FAISS vector store."""
        reader = PdfReader(pdf_file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        docs = [Document(page_content=text)]

        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(docs, embeddings)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=False
        )

    def ask(self, query: str) -> str:
        """Ask question to LLM."""
        if not self.chain:
            return "Please upload a PDF first."

        output = self.chain({"question": query})
        return output.get("answer", "I could not find an answer.")
