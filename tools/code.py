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
        """ Extracts the code excluding everything after 'output' word.
        Tries to get all chunks, but also returns only the first chunk.
        This is due to how the LLMs tend to structure code outputs"""
        # Recortar si el codigo contiene la palabra "output" ya que el LLM
        # tiende a poner un ejemplo de como sera el output pero eso
        # no tiene codigo y da error, es mejor borrarlo
        if text.count("```") >= 2:
            first = text.find("```")
            second = text.find("```", first + 3)
            output_index = text.find("output", second + 3)
            if output_index != -1:
                text = text[:output_index]
        print(text)
        matches = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
        return ("\n\n".join(match.strip() for match in matches), matches[0].strip()) if matches else (None, None)


    def execute_code(code, first_code_block):
        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(code)
                captured_output = output_buffer.getvalue()
            return True, captured_output
        except Exception as e1:
            output_buffer = io.StringIO()  # Reset buffer
            try:
                with contextlib.redirect_stdout(output_buffer):
                    exec(first_code_block)
                    captured_output = output_buffer.getvalue()
                return True, captured_output
            except Exception as e2:
                return False, f"Primary execution failed: {e1}\nFallback also failed: {e2}"
        
    
    original_query = query
    retries = 0
    while retries < max_retries:
        logger.info("Generating code")
        raw_code = generate_code(query)
        code, first_code = extract_code(raw_code)
        print("code")
        print(code)
        if code:
            print("\033[35m" +  "CODE to be RUN\n" + code + "\033[0m")
            ok, output = execute_code(code, first_code)
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