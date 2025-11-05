# app/__init__.py

"""
app package for the Generative AI Chat Assistant.

Contains modules:
- pdf_loader: load & index PDFs into a vector store
- chat_engine: build a conversational retrieval chain
- memory_store: simple memory wrapper (optional persistence)
"""
__all__ = ["pdf_loader", "chat_engine", "memory_store"]
