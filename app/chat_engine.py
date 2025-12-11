# app/chat_engine.py

from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS


class ChatEngine:
    """Wrapper class for LangChain’s ConversationalRetrievalChain."""

    def __init__(self, vectorstore: FAISS, model_name="gpt-4o-mini"):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True
        )

    def ask(self, query: str, chat_history=None) -> str:
        """Ask question to LLM and return answer only."""

        inputs = {"question": query}

        if chat_history:
            inputs["chat_history"] = chat_history

        output = self.chain(inputs)
        answer = output.get("answer", "Sorry, I couldn't find an answer.")

        return answer
