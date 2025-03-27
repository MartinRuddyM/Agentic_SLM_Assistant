from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
import yaml
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from create_db import EmbeddingDB
from conversation import Conversation
from tools.llm_functions import *

# API key setups
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# LLm Chat setups
default_chat = ChatMistralAI(
    model="open-mistral-7b",
    api_key=MISTRAL_API_KEY
)

cheap_chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    google_api_key=GEMINI_API_KEY
)

embeddings_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Prompt setups
with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

def main():
    db = EmbeddingDB(db_path="db/user_data.db", faiss_path="db/faiss_index.index")
    conversation = Conversation(user_id, default_chat, cheap_chat, prompts)
    print("Type exit to exit the conversation")
    while True:
        user_input = input("Type question: ")
        if len(user_input) == 4 and user_input.lower() == "exit":
            break
        # Procesar la query con ReAct framework, dando acceso a las herramientas disponibles
        # Al procesar, introudcir info relevante del usuario y de conversaciones pasadas si procede
    final_summary, user_info = conversation.exit_conversation()
    db.add_conversation_summary(user_id=user_id, summary_text=final_summary)
    # Comprobar que la user info sea realmente nueva, para ello hay que anadir una funcion de llm que devuelva solo los statements que osn realmente nuevos
    db.close_db()
    


if __name__ == "__main__":
    main()



# Siguientes pasos> Implementar el centro que son los pasos ReAct para llegar a respuestas. Luego, ver como insertar la info pasada. Luego, implementar funciones de chequeo que queden, como la de info del usuario nueva
# Despues testear todo el sistema y presentarselo a Carlos
# Puedo anadirle una interfaz basica que le dara muchos mas puntos.