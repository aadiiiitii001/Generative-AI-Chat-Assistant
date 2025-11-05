# app/main.py
from chat_engine import ChatEngine  # or your main chat class
import streamlit as st

# Initialize your chat assistant
chat = ChatEngine()

st.title("Generative AI Chat Assistant")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    chat.load_pdf(uploaded_file)
    query = st.text_input("Ask a question:")
    if query:
        answer = chat.ask(query)
        st.write(answer)
