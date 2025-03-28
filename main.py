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
        prompts = yaml.safe_load(file)

    # Load DB and start processing
    db = EmbeddingDB(db_path="db/user_data.db", faiss_path="db/faiss_index.index")
    conversation = Conversation(default_chat, cheap_chat, prompts)
    debug = False
    while True:
        query = input("Type question: ")
        if len(query) == 4 and query.lower() == "exit":
            break
        # Procesar la query con ReAct framework, dando acceso a las herramientas disponibles
        # Al procesar, introudcir info relevante del usuario y de conversaciones pasadas si procede
        relevant_user_info = db.search(query, source="user_info")
        relevant_past_conversations = db.search(query, source="conversation")
        prompt = prepare_prompt(relevant_user_info, relevant_past_conversations, query)
        if debug:
            steps, answer = ReAct_process(llm=default_chat, query=prompt, prompts=prompts, debug=True)
            print(steps)
        else:
            answer = ReAct_process(llm=default_chat, query=prompt, prompts=prompts)
        # Acabar de procesar mediante una query llm para personalizarlo al usuario.
        final_answer = personalize_final_answer(answer, conversation.conversation_history, default_chat, prompts)
        print(final_answer)
        conversation.add_interaction(query, final_answer)

    if len(conversation.conversation_history) != 0:
        final_summary, user_info = conversation.exit_conversation()
        db.add_conversation_summary(summary_text=final_summary)
        # Comprobar que la user info sea realmente nueva, para ello hay que anadir una funcion de llm que devuelva solo los statements que osn realmente nuevos
    db.close_db()
    


if __name__ == "__main__":
    main()



# Siguientes pasos> Implementar el centro que son los pasos ReAct para llegar a respuestas. Luego, ver como insertar la info pasada. Luego, implementar funciones de chequeo que queden, como la de info del usuario nueva
# Despues testear todo el sistema y presentarselo a Carlos
# Puedo anadirle una interfaz basica que le dara muchos mas puntos.