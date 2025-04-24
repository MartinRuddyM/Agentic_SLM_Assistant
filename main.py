from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
import yaml
import os
from dotenv import load_dotenv

from manage import EmbeddingDB, prepare_prompt
from ReAct import ReAct_process
from conversation import Conversation
from tools.llm_functions import *
from logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Starting system...")
    load_dotenv()
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    default_chat = ChatMistralAI(
        model="open-mistral-7b",
        #model="mistral-large-latest",
        api_key=MISTRAL_API_KEY
    )
    cheap_chat = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        google_api_key=GEMINI_API_KEY
    )

    with open("prompts.yaml", "r") as file:
        system_prompts = yaml.safe_load(file)

    available_tools = ["Web Search", "Run Code"]

    logger.info("Loading DB...")
    db = EmbeddingDB(db_path="db/user_data.db", faiss_conversation_path="db/faiss_conversations.index", faiss_user_info_path="db/faiss_user_info.index")
    conversation = Conversation(default_chat, cheap_chat, system_prompts)
    debug = False
    while True:
        query = input("Type question: ")
        if len(query) == 4 and query.lower() == "exit":
            logger.info("Exiting conversation...")
            break
        relevant_user_info = db.search(query, source="user_info", top_k=20)
        relevant_past_conversations = db.search(query, source="conversation")
        #prompt = prepare_prompt(conversation, relevant_user_info, query, system_prompts)
        react_task_desc = get_react_task_desc(relevant_user_info, relevant_past_conversations, conversation, available_tools, query, default_chat, system_prompts)
        ############
        ####Queda por hacer, cambiar la prompt de task desc para que sea mucho mas
        # corto lo que anada y solo si hay algo de contexto importante, si no no.
        answer = ReAct_process(query, react_task_desc, system_prompts, default_chat, cheap_chat)
        final_answer = personalize_final_answer(query, answer, relevant_user_info, relevant_past_conversations, conversation, default_chat, system_prompts)
        logger.info("Final answer generated.")
        print(final_answer)
        conversation.add_interaction(query, final_answer)

    if len(conversation.history) > 0:
        final_summary, new_user_info = conversation.exit_conversation()
        db.add_conversation_summary(final_summary)
        existing_user_info = db.get_all_user_information()
        if len(existing_user_info) > 0:
            new_user_info = contrast_user_information(existing_user_info, new_user_info)
        db.add_user_info(new_user_info)
        # Este paso se podria mejorar para que, ademas de detectar la info nueva, si hay info contradictoria modificara la BBDD para dejar la mas reciente
        # Tambien estaria bien una posibilidad del usuario de revisar su info para modificarla o borrarla manualmente
        # Un posible problema es que las informaciones del usuario crezcan indefinidamente
    db.close_db()
    logger.info("Database closed.")
    


if __name__ == "__main__":
    main()
