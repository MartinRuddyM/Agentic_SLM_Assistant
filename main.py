from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
import yaml
import os
from datetime import datetime, timedelta

from conversation import Conversation

# API key setups
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPLAKE_API_KEY = os.getenv("DEEPLAKE_API_KEY")
DEEPLAKE_USERNAME = os.getenv("DEEPLAKE_USERNAME")

# LLm Chat setups
default_chat = ChatMistralAI(
    model="open-mistral-7b",
    api_key=MISTRAL_API_KEY
)

cheap_chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    google_api_key=GEMINI_API_KEY
)

# Prompt setups
with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

# Global variables
user_id = '123'


def main():
    # Update database, deleting old conversations
    max_conversation_days = 30


    conversation = Conversation(user_id, default_chat, cheap_chat, prompts)
    print("Type exit to exit the conversation")
    while True:
        user_input = input("Type question: ")
        if len(user_input) == 4 and user_input.lower() == "exit":
            break


if __name__ == "__main__":
    main()