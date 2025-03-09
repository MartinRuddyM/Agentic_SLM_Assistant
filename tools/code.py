import re
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def generate_code(query):
    mistral_chat = ChatMistralAI(model="mistral-large-2411", api_key=MISTRAL_API_KEY)
    system_prompt = "You are an AI that generates Python code. Omit explanations. Respond with only the Python code needed for the query. PRINT the results. Query: "
    return mistral_chat.predict(system_prompt + " " + query)

def extract_code(text):
    matches = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
    return "\n\n".join(match.strip() for match in matches) if matches else None

def execute_code(code):
    print(code)
    try:
        exec(code, globals())
    except Exception as e:
        print(f"Execution failed: {e}")
        return str(e)
    return None

def run(query, max_retries=3):
    original_query = query
    retries = 0
    while retries < max_retries:
        result = generate_code(query)
        code = extract_code(result)
        if code:
            error_message = execute_code(code)
            if not error_message:
                return
            print(f"Error running previous code:\n====\n{code}\n====\nError:\n====\n{error_message}\n====\nRetrying with improved prompt...\n")
            query = f"The previous code failed with error: {error_message}.\n\n Previous code: {code}\n\n Solve the error and generate correct code"
        else:
            print("No valid Python code found, retrying...")
        retries += 1
    print("Max retries reached. Unable to generate working code.")

run("Given an A/B experiment, perform a one-way ANOVA on this data: method_A = [85, 90, 88]  method_B = [78, 82, 80]. Show if there is significative difference")