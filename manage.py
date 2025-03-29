import sqlite3
import os
import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from typing import Literal
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from tools.code import run_code
from tools.web_search import web_search


class EmbeddingDB:
    def __init__(self, db_path, faiss_conversation_path, faiss_user_info_path, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.faiss_conversation_path = faiss_conversation_path
        self.faiss_user_info_path = faiss_user_info_path
        self.model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        # Comprueba si existen las BBDD y si no, las crea.
        files = [db_path, faiss_conversation_path, faiss_user_info_path]
        existing = [os.path.exists(f) for f in files]
        if all(existing):
            pass
        elif not any(existing):
            self._create_db_files()
        else:
            raise FileNotFoundError("Inconsistent state: some required files are missing")
        
        self.conn = sqlite3.connect(self.db_path)
        self.faiss_conversations = faiss.read_index(self.faiss_conversation_path)
        self.faiss_user_info = faiss.read_index(self.faiss_user_info_path)


    def _create_db_files(self):
        print("Creating DB files, this might take a while...")
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

        print("Successfully created DB files")


    def close_db(self):
        #faiss.write_index(self.faiss_conversations, self.faiss_conversation_path)
        #faiss.write_index(self.faiss_user_info, self.faiss_user_info_path)
        self.conn.close()


    def add_user_info(self, texts: list[str]):
        if len(texts) == 0:
            return
        cursor = self.conn.cursor()
        embeddings = []
        ids = []
        for text in texts:
            cursor.execute("INSERT INTO user_info (text) VALUES (?)", (text,))
            info_id = cursor.lastrowid
            embedding = self.model.encode(text)
            embeddings.append(embedding)
            ids.append(info_id)
        
        self.conn.commit()
        embedding_np = np.array(embeddings).astype("float32")
        ids_np = np.array(ids)
        self.faiss_user_info.add_with_ids(embedding_np, ids_np)
        faiss.write_index(self.faiss_user_info, self.faiss_user_info_path)


    def add_conversation_summary(self, text: str):
        if not text:
            return
        embedding = self.model.encode(text)
        embedding_np = np.array([embedding]).astype("float32")
        date_str = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO conversation (text, date) VALUES (?, ?)", (text, date_str))
        self.conn.commit()
        summary_id = cursor.lastrowid
        self.faiss_conversations.add_with_ids(embedding_np, np.array([summary_id]))
        faiss.write_index(self.faiss_conversations, self.faiss_conversation_path)


    def search(self, query, source: Literal["conversation", "user_info"]="conversation", top_k=5):
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

    
    def get_all_user_information(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT text FROM user_info")
        rows = cursor.fetchall()
        return [row[0] for row in rows] if rows else []


def prepare_prompt(conversation, relevant_user_info, query, prompts):
    """Given relevant contextual user info and past conversations, prepare the prompt
    to be fed into ReAct."""
    summary_last_messages = conversation.get_last_n_summaries(n=5)
    user_info = "\n".join(text[0] for text in relevant_user_info)
    values = {
        "query":query,
        "past_messages":summary_last_messages,
        "user_info":user_info,
    }
    final_prompt = prompts["feed_to_react"].format(**values)
    return final_prompt



def ReAct_process(query:str, prompts, llm, summarizer_llm, max_iter=10, debug=False):
    def run_code_wrapper(task_desc):
        return run_code(task_desc, llm, prompts)
    
    def web_search_wrapper(search_terms):
        return web_search(search_terms, query, prompts, summarizer_llm)
    
    tools = [
        Tool(name="Web Search", func=web_search_wrapper, description=prompts["web_search_tool_description"]),
        Tool(name="Run Code", func=run_code_wrapper, description=prompts["code_tool_description"]),
    ]
    #####################
    ############TODO
    #####################
    # Adjust react prompt so it makes sense with my custom system prompt and I dont have to duplicate
    prompt = PromptTemplate.from_template(prompts["react_prompt"])
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=debug, max_iterations=max_iter, max_execution_time=180, early_stopping_method="generate", handle_parsing_errors=True)
    response = agent_executor.invoke({"input":query})
    print(response)
    final_answer = response["output"]
    intermediate_steps = "\n".join(f"{action.log}\nObservation: {observation}" for action, observation in response["intermediate_steps"])
    if debug:
        return intermediate_steps, final_answer
    return None, final_answer

    