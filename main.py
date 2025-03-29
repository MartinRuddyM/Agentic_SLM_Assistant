from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
import yaml
import os
from dotenv import load_dotenv

from manage import EmbeddingDB, prepare_prompt, ReAct_process
from conversation import Conversation
from tools.llm_functions import *

def main():
    # API keys
    load_dotenv()
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Prepare LLM models
    default_chat = ChatMistralAI(
        model="open-mistral-7b",
        api_key=MISTRAL_API_KEY
    )
    cheap_chat = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        google_api_key=GEMINI_API_KEY
    )
    #embeddings_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Load system prompts
    with open("prompts.yaml", "r") as file:
        system_prompts = yaml.safe_load(file)

    # Load DB and start processing
    db = EmbeddingDB(db_path="db/user_data.db", faiss_path="db/faiss_index.index")
    conversation = Conversation(default_chat, cheap_chat, system_prompts)
    debug = False
    while True:
        query = input("Type question: ")
        if len(query) == 4 and query.lower() == "exit":
            break
        # Informacion de contexto del usuario
        relevant_user_info = db.search(query, source="user_info", top_k=20)
        prompt = prepare_prompt(conversation, relevant_user_info, query, system_prompts)
        if debug:
            steps, answer = ReAct_process(llm=default_chat, query=prompt, prompts=system_prompts, debug=True)
            print(steps)
        else:
            answer = ReAct_process(llm=default_chat, query=prompt, prompts=system_prompts)
        # Acabar de procesar mediante una query llm para personalizarlo mas al usuario.
        relevant_past_conversations = db.search(query, source="conversation")
        final_answer = personalize_final_answer(query, answer, relevant_user_info, relevant_past_conversations, conversation, default_chat, system_prompts)
        print(final_answer)
        conversation.add_interaction(query, final_answer)

    if len(conversation.conversation_history) > 0:
        final_summary, new_user_info = conversation.exit_conversation()
        db.add_conversation_summary(summary_text=final_summary)
        existing_user_info = db.get_all_user_information()
        if len(existing_user_info) > 0:
            new_user_info = contrast_user_information(existing_user_info, new_user_info)
        # Este paso se podria mejorar para que, ademas de detectar la info nueva, si hay info contradictoria modificara la BBDD para dejar la mas reciente
        # Tambien estaria bien una posibilidad del usuario de revisar su info para modificarla o borrarla manualmente
        # Un posible problema es que las informaciones del usuario crezcan indefinidamente
        db.add_user_info(new_user_info)
    db.close_db()
    


if __name__ == "__main__":
    main()



# Siguientes pasos> Implementar el centro que son los pasos ReAct para llegar a respuestas. Luego, ver como insertar la info pasada. Luego, implementar funciones de chequeo que queden, como la de info del usuario nueva
# Despues testear todo el sistema y presentarselo a Carlos
# Puedo anadirle una interfaz basica que le dara muchos mas puntos.



#TODO
# Cambiar el sistema de resumenes para que no resuma de 5 en 5 sino de 1 en 1, y ajustar las funciones que lo usan actualmente
