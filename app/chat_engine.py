# app/chat_engine.py

from typing import Any, Dict, Optional

# ✅ Correct imports for new LangChain versions
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS


def build_chat_chain(
    vectorstore: FAISS,
    llm_model_name: str = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> ConversationalRetrievalChain:
    """
    Build ConversationalRetrievalChain with FAISS vectorstore.
    """

    model_name = llm_model_name or "gpt-4o-mini"

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain


def ask_chain(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Execute query against chat chain.
    """

    inputs = {"question": question}

    if chat_history:
        inputs["chat_history"] = chat_history

    output = chain(inputs)

    return {
        "answer": output.get("answer"),
        "source_documents": output.get("source_documents", []),
        "raw": output,
    }
