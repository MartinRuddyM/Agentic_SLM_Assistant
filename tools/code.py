import re
from dotenv import load_dotenv
import os
import io
import contextlib
from logger import get_logger

logger = get_logger(__name__)

def run_code(query, llm, prompts, max_retries=3):
    """Takes an instruction with the descriptions of what code to run and the original user
    query for context. Then, the LLM here writes code to solve the problem and prints its output.
    The output is returned as text to the ReAct system.
    If it was not possible to run code, an error message is returned instead."""

    logger.info("Called Tool: Code")

    def generate_code(query):
        values = {
            "task_description":query,
        }
        final_prompt = prompts["code_task_description"].format(**values)
        return llm.invoke(final_prompt).content

    def extract_code(text):
        print(text)
        matches = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
        return "\n\n".join(match.strip() for match in matches) if matches else None

    def execute_code(code):
        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(code)
                captured_output = output_buffer.getvalue()
        except Exception as e:
            #print(f"Execution failed: {e}")
            return False, str(e)
        return True, captured_output
    
    original_query = query
    retries = 0
    while retries < max_retries:
        logger.info("Generating code")
        raw_code = generate_code(query)
        code = extract_code(raw_code)
        if code:
            ok, output = execute_code(code)
            if ok:
                logger.info("Code ran successfully, returning")
                return "The output from the code was the following: " + output
            #print(f"Error running previous code:\n====\n{code}\n====\nError:\n====\n{error_message}\n====\nRetrying with improved prompt...\n")
            query = f"The previous code failed with error: {output}.\n\n Previous code: {code}\n\nOriginal problem to solve: {original_query}\n\n Solve the error and generate correct code"
        else:
            print("No valid Python code found, retrying...")
        logger.info("Code failed, retrying")
        retries += 1
    logger.info("Code Tool failed, no more retries left.")
    return prompts["code_tool_error"]

if __name__ == "__main__":
    # Sample query to test the working
    #print("Given an A/B experiment, perform a one-way ANOVA on this data: method_A = [85, 90, 88]  method_B = [78, 82, 80]. Show if there is significative difference")

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_mistralai import ChatMistralAI
    import yaml
    import os
    from dotenv import load_dotenv

    # API key setups
    load_dotenv()
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

    with open("prompts.yaml", "r") as file:
        prompts = yaml.safe_load(file)

    default_chat = ChatMistralAI(
        model="open-mistral-7b",
        api_key=MISTRAL_API_KEY
    )

    query = "Make code to compute the following equation: ((123 * 300) - 200 + 45) / 33. Make sure to print ONLY UP TO 3 decimals"

    print(1)
    output = run_code(query, default_chat, prompts)
    print(output)