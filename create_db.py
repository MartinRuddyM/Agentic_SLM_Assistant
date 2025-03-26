import sqlite3
import json
import os
import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

class EmbeddingDB:
    def __init__(self, db_path, faiss_conversation_path, faiss_user_info_path, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", create_db_files=False):
        self.db_path = db_path
        self.faiss_conversation_path = faiss_conversation_path
        self.faiss_user_info_path = faiss_user_info_path
        self.model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        if create_db_files:
            self._create_db_files()
        else:
            if not all(os.path.exists(p) for p in [db_path, faiss_conversation_path, faiss_user_info_path]):
                raise FileNotFoundError("One or more required files do not exist and create_db_files=False")
        
        self.conn = sqlite3.connect(self.db_path)
        self.faiss_conversations = faiss.read_index(self.faiss_conversation_path)
        self.faiss_user_info = faiss.read_index(self.faiss_user_info_path)

    def _create_db_files(self):
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    date TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT
                )
            ''')
            conn.commit()
            conn.close()

        if not os.path.exists(self.faiss_conversation_path):
            conv_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
            faiss.write_index(conv_index, self.faiss_conversation_path)

        if not os.path.exists(self.faiss_user_info_path):
            user_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
            faiss.write_index(user_index, self.faiss_user_info_path)

    def close_db(self):
        faiss.write_index(self.faiss_conversations, self.faiss_conversation_path)
        faiss.write_index(self.faiss_user_info, self.faiss_user_info_path)
        self.conn.close()

    def add_user_info(self, text):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO user_info (text) VALUES (?)", (text,))
        self.conn.commit()
        info_id = cursor.lastrowid
        embedding = self.model.encode(text)
        embedding_np = np.array([embedding]).astype("float32")
        self.faiss_user_info.add_with_ids(embedding_np, np.array([info_id]))
        faiss.write_index(self.faiss_user_info, self.faiss_user_info_path)

    def add_conversation_summary(self, text):
        embedding = self.model.encode(text)
        embedding_np = np.array([embedding]).astype("float32")
        date_str = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO conversation (text, date) VALUES (?, ?)", (text, date_str))
        self.conn.commit()
        summary_id = cursor.lastrowid
        self.faiss_conversations.add_with_ids(embedding_np, np.array([summary_id]))
        faiss.write_index(self.faiss_conversations, self.faiss_conversation_path)

    def search(self, query, source="conversation", top_k=5):
        query_emb = self.model.encode(query).astype("float32")
        faiss_index = self.faiss_conversations if source == "conversation" else self.faiss_user_info
        D, I = faiss_index.search(np.array([query_emb]), k=top_k)
        results = []
        cursor = self.conn.cursor()
        for idx in I[0]:
            if idx == -1:
                continue
            if source == "conversation":
                cursor.execute("SELECT text, date FROM conversation WHERE id = ?", (int(idx),))
            else:
                cursor.execute("SELECT text FROM user_info WHERE id = ?", (int(idx),))
            row = cursor.fetchone()
            if row:
                results.append(row)
        return results

    def delete_old_conversations(self, max_conversation_days=30):
        cutoff_date = (datetime.today() - timedelta(days=max_conversation_days)).strftime('%Y-%m-%d')
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM conversation WHERE date < ?", (cutoff_date,))
        ids_to_delete = [row[0] for row in cursor.fetchall()]
        if ids_to_delete:
            id_array = np.array(ids_to_delete, dtype=np.int64)
            self.faiss_conversations.remove_ids(id_array)
        cursor.execute("DELETE FROM conversation WHERE date < ?", (cutoff_date,))
        self.conn.commit()


if __name__ == "__main__":
    new_db = EmbeddingDB(
        db_path="db/user_data.db",
        faiss_path="db/faiss_index.index",
        create_db_files=True
    )