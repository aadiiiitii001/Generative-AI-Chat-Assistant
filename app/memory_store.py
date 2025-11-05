# app/memory_store.py

import json
import os
from typing import List, Dict, Optional

DEFAULT_MEMORY_DIR = "memory"

class ChatMemory:
    """
    Simple in-process memory store for conversation history.
    Each message is stored as a dict: {"role": "user"/"assistant", "content": "..."}

    This is intentionally simple so you can swap it out with Redis, DB, or LangChain memory later.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        os.makedirs(DEFAULT_MEMORY_DIR, exist_ok=True)
        self.filepath = os.path.join(DEFAULT_MEMORY_DIR, f"{self.session_id}.json")
        self.history: List[Dict[str, str]] = self._load_from_disk()

    def _load_from_disk(self) -> List[Dict[str, str]]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _persist(self):
        try:
            with open(self.filepath, "w", encoding="utf8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception:
            # swallow errors for now (optional: log)
            pass

    def add_user_message(self, text: str):
        self.history.append({"role": "user", "content": text})
        self._persist()

    def add_assistant_message(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self._persist()

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def clear(self):
        self.history = []
        self._persist()
