import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory


class SnapshotManager:
    def __init__(self, snapshot_dir: str = "./snapshots"):
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.session_id = None
        self.session_path = None
        self.history = []
        self.alias_file = os.path.join(self.snapshot_dir, "aliases.json")
        self.aliases = self._load_aliases()
        
    def _load_aliases(self):
        if os.path.exists(self.alias_file):
            with open(self.alias_file, "r") as f:
                return json.load(f)
        return {}

    def _save_aliases(self):
        with open(self.alias_file, "w") as f:
            json.dump(self.aliases, f, indent=2)
            

    def start_new_session(self, alias: Optional[str] = None) -> str:
        self.session_id = str(uuid.uuid4())
        self.session_path = os.path.join(self.snapshot_dir, f"{self.session_id}.json")
        self.history = []
        
        self.aliases[self.session_id] = self.session_id
        if alias:
            self.aliases[alias] = self.session_id
        self._save_aliases()
        return self.session_id

    def resume_session(self, identifier: str) -> Optional[ConversationBufferMemory]:
        session_id = self.aliases.get(identifier, identifier)
        self.session_id = session_id
        self.session_path = os.path.join(self.snapshot_dir, f"{session_id}.json")
        
        if not os.path.exists(self.session_path):
            print("❌ Session not found.")
            return None

        try:
            with open(self.session_path, "r") as f:
                self.history = json.load(f)
        except Exception:
            print("⚠️ Failed to load session history.")
            return None

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for item in self.history:
            memory.chat_memory.add_user_message(HumanMessage(content=item["question"]))
            memory.chat_memory.add_ai_message(AIMessage(content=item["answer"]))
        return memory
    
    def resume_latest(self) -> Optional[ConversationBufferMemory]:
        sessions = self.list_sessions()
        if not sessions:
            print("❌ No sessions found.")
            return None
        return self.resume_session(sessions[0]["id"])
    
    def list_sessions(self) -> List[Dict]:
        sessions = []
        for file in os.listdir(self.snapshot_dir):
            if file.endswith(".json") and file != "aliases.json":
                session_id = file.replace(".json", "")
                path = os.path.join(self.snapshot_dir, file)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    first_msg = data[0]["question"] if data else ""
                except Exception:
                    first_msg = "(corrupt or empty)"
                timestamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
                alias = self._get_alias_for_id(session_id)
                sessions.append({
                    "id": session_id,
                    "alias": alias,
                    "timestamp": timestamp,
                    "first_msg": first_msg
                })
        return sorted(sessions, key=lambda s: s["timestamp"], reverse=True)
    
    def _get_alias_for_id(self, session_id: str) -> str:
        # Reverse lookup alias pointing to session_id
        for alias, sid in self.aliases.items():
            if sid == session_id and alias != session_id:
                return alias
        return session_id  # fallback to ID if no custom alias

    def record_turn(self, question: str, answer: str, sources: List[Dict]):
        self.history.append({
            "question": question,
            "answer": answer,
            "sources": sources
        })

    def save_snapshot(self):
        if not self.session_path:
            self.session_path = os.path.join(self.snapshot_dir, f"{self.session_id}.json")
        with open(self.session_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 Snapshot saved to: {self.session_path}")
