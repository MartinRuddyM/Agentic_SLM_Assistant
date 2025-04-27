from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
import yaml
import os
from dotenv import load_dotenv

from tools.manage import EmbeddingDB
from ReAct import ReAct_process
from conversation import Conversation
from tools.llm_functions import *
from logger import get_logger

logger = get_logger(__name__)

class AppContext:
    def __init__(self):
        logger.info("Starting system...")
        load_dotenv()

        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

        chat1 = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY
        )
        chat2 = ChatMistralAI(
            model="open-mistral-7b",
            api_key=MISTRAL_API_KEY
        )
        chat3 = ChatGroq(
            model="llama3-8b-8192",
            api_key=GROQ_API_KEY
        )
        chat4 = ChatMistralAI(
            model="mistral-large-latest",
            api_key=MISTRAL_API_KEY
        )

        self.default_chat = chat2
        self.cheap_chat = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-8b",
            google_api_key=GEMINI_API_KEY
        )

        with open("prompts.yaml", "r") as file:
            self.system_prompts = yaml.safe_load(file)

        logger.info(f"Loading DB...")
        self.db = EmbeddingDB(
            db_path="db/user_data.db",
            faiss_conversation_path="db/faiss_conversations.index",
            faiss_user_info_path="db/faiss_user_info.index"
        )
        logger.info(f"DB Loaded succesfully")

        self.conversation = Conversation(
            self.default_chat,
            self.cheap_chat,
            self.system_prompts
        )

    def handle_query(self, query: str) -> dict:
        logger.info(f"Received query: {query}")
        relevant_user_info = self.db.search(query, source="user_info", top_k=20)
        relevant_past_conversations = self.db.search(query, source="conversation")

        react_task_desc = get_react_task_desc(
            relevant_user_info, relevant_past_conversations, self.conversation,
            query, self.cheap_chat, self.system_prompts
        )
        answer, reasoning = ReAct_process(query, react_task_desc, self.conversation, self.system_prompts, self.default_chat, self.cheap_chat)
        final_answer = personalize_final_answer(query, answer, relevant_user_info, relevant_past_conversations, self.conversation, self.default_chat, self.system_prompts)

        self.conversation.add_interaction(query, final_answer, reasoning)
        logger.info("Final answer generated.")

        return {"final_answer": final_answer, "reasoning": reasoning}

    def exit_session(self):
        if len(self.conversation.history) > 0:
            final_summary, new_user_info = self.conversation.exit_conversation()
            self.db.add_conversation_summary(final_summary)
            existing_user_info = self.db.get_all_user_information()
            if len(existing_user_info) > 0:
                new_user_info = contrast_user_information(
                    existing_user_info, new_user_info,
                    self.cheap_chat, self.system_prompts
                )
            self.db.add_user_info(new_user_info)
        self.db.close_db()
        logger.info("Database closed.")
