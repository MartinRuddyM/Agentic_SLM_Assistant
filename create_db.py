import sqlite3
import json
import os
import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

class EmbeddingDB:
    def __init__(self, db_path, faiss_path, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", create_db_files=False):
        self.db_path = db_path
        self.faiss_path = faiss_path
        self.model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        if create_db_files:
            self._create_db_files()
        else:
            if not os.path.exists(db_path) or not os.path.exists(faiss_path):
                raise FileNotFoundError("DB or FAISS file does not exist and create_db_files=False")

        self.conn = sqlite3.connect(self.db_path)
        self.faiss_index = faiss.read_index(self.faiss_path)

    def _create_db_files(self):
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS permanent_user_info (
                    user_id TEXT PRIMARY KEY,
                    user_info TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    user_info TEXT,
                    date TEXT,
                    FOREIGN KEY (user_id) REFERENCES permanent_user_info(user_id)
                )
            ''')
            conn.commit()
            conn.close()
        if not os.path.exists(self.faiss_path):
            index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
            faiss.write_index(index, self.faiss_path)

    def close_db(self):
        faiss.write_index(self.faiss_index, self.faiss_path)
        self.conn.close()

    def add_user_info(self, user_id, user_info_list):
        conn = self.conn
        cursor = conn.cursor()
        user_info_json = json.dumps(user_info_list)
        cursor.execute(
            "INSERT OR REPLACE INTO permanent_user_info (user_id, user_info) VALUES (?, ?)",
            (user_id, user_info_json)
        )
        conn.commit()

    def add_conversation_summary(self, user_id, summary_text):
        conn = self.conn
        cursor = conn.cursor()
        embedding = self.model.encode(summary_text)
        date_str = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO conversation_summaries (user_id, user_info, date) VALUES (?, ?, ?)",
            (user_id, summary_text, date_str)
        )
        conn.commit()
        summary_id = cursor.lastrowid
        embedding_np = np.array([embedding]).astype("float32")
        self.faiss_index.add_with_ids(embedding_np, np.array([summary_id]))
        faiss.write_index(self.faiss_index, self.faiss_path)

    def search(self, query, top_k=5):
        query_emb = self.model.encode(query).astype("float32")
        D, I = self.faiss_index.search(np.array([query_emb]), k=top_k)
        conn = self.conn
        cursor = conn.cursor()
        results = []
        for db_id in I[0]:
            if db_id == -1:
                continue
            cursor.execute("SELECT user_id, user_info, date FROM conversation_summaries WHERE id=?", (int(db_id),))
            row = cursor.fetchone()
            if row:
                results.append(row)
        return results

    def delete_old_conversations(self, max_conversation_days=30):
        """Removes old conversations given the days past, from both normal and FAISS DBs"""
        cutoff_date = (datetime.today() - timedelta(days=max_conversation_days)).strftime('%Y-%m-%d')
        conn = self.conn
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM conversation_summaries WHERE date < ?",
            (cutoff_date,)
        )
        ids_to_delete = [row[0] for row in cursor.fetchall()]
        if ids_to_delete:
            id_array = np.array(ids_to_delete, dtype=np.int64)
            self.faiss_index.remove_ids(id_array)
        cursor.execute(
            "DELETE FROM conversation_summaries WHERE date < ?",
            (cutoff_date,)
        )
        conn.commit()


if __name__ == "__main__":
    new_db = EmbeddingDB(
        db_path="db/user_data.db",
        faiss_path="db/faiss_index.index",
        create_db_files=True
    )