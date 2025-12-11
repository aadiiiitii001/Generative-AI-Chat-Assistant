from chat_engine import ChatEngine
import streamlit as st

chat = ChatEngine()

st.title("Generative AI Chat Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully! Building vector index...")
    chat.load_pdf(uploaded_file)
    st.success("PDF indexed. You can now ask questions.")

query = st.text_input("Ask a question:")

if query:
    if chat.index is None:
        st.warning("Please upload a PDF first.")
    else:
        answer = chat.ask(query)
        st.write(answer)
