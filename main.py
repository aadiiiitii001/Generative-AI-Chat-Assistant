import streamlit as st
from app.chat_engine import ChatEngine
from app.pdf_loader import load_pdfs
from app.memory_store import get_memory

st.set_page_config(page_title="Generative AI Chat Assistant", layout="wide")

st.title("🤖 Generative AI Chat Assistant")
st.markdown("Ask questions about your company policies or uploaded documents!")

# Load vector store / embeddings from PDFs
vectorstore = load_pdfs("app/data")

# Initialize chat engine
chat_engine = ChatEngine(vectorstore)
memory = get_memory()

# Chat interface
user_input = st.text_input("You:", "")

if st.button("Ask") and user_input:
    response = chat_engine.ask(user_input, memory)
    st.markdown(f"**Assistant:** {response}")
