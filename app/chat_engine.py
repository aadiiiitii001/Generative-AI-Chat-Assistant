# app/chat_engine.py

from typing import Any, Dict, Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.schema import Document

def build_chat_chain(
    vectorstore: FAISS,
    llm_model_name: str = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> ConversationalRetrievalChain:
    """
    Build and return a ConversationalRetrievalChain using the provided vectorstore.
    - vectorstore: FAISS instance from pdf_loader.create_vectorstore_from_pdf
    - llm_model_name: optional model name (e.g., 'gpt-4o-mini' or 'gpt-3.5-turbo')
    """
    # Choose LLM - default to gpt-3.5-turbo if not provided
    model_name = llm_model_name or "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True  # useful for showing sources/traceability
    )

    return chain


def ask_chain(chain: ConversationalRetrievalChain, question: str, chat_history: Optional[list] = None) -> Dict[str, Any]:
    """
    Ask a question using the chain. Returns a dict with:
    - answer: str
    - source_documents: list[Document]
    - chat_history: updated memory (if supported)
    """
    # chain expects {"question": "..."} for ConversationalRetrievalChain
    inputs = {"question": question}
    if chat_history:
        # some chains allow passing chat_history; many rely on memory
        inputs["chat_history"] = chat_history

    output = chain(inputs)
    answer = output.get("answer")
    sources = output.get("source_documents", [])
    # Optionally extract the chat history from the memory object if present
    return {"answer": answer, "source_documents": sources, "raw": output}
