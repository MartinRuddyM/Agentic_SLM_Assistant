from tools.code import run_code
from tools.web_search import web_search
from logger import get_logger
import re
from typing import List

logger = get_logger(__name__)

def ReAct_process(query:str, prompts:List[str], llm, summarizer_llm, max_iter=10):

    def set_up_tools():
        def run_code_wrapper(task_desc):
            return run_code(task_desc, llm, prompts)

        def web_search_wrapper(search_terms):
            return web_search(search_terms, query, prompts, summarizer_llm)

        tools = {
            "Web Search": {
                "func": web_search_wrapper,
                "description": prompts["web_search_tool_description"]
            },
            "Run Code": {
                "func": run_code_wrapper,
                "description": prompts["code_tool_description"]
            }
        }
        return tools


    def build_react_prompt(agent_scratchpad, tools):
        tool_descriptions = "\n".join(
            f"{name}: {tool['description']}" for name, tool in tools.items()
        )
        tool_names = ", ".join(tools.keys())
        values = {
            "tool_descriptions":tool_descriptions,
            "tool_names":tool_names,
            "input":query,
            "agent_scratchpad":agent_scratchpad,
        }
        return prompts["react_prompt"].format(**values)
    #####################
    ############TODO
    #####################
    # Adjust react prompt so it makes sense with my custom system prompt and I dont have to duplicate


    def parse_react_output(output):
        action_match = re.search(r"Action: (.*)\nAction Input: (.*)", output)
        final_answer_match = re.search(r"Final Answer: (.*)", output)
        thought_match = re.search(r"Thought:(.*)", output)

        if final_answer_match:
            return {"type": "final", "answer": final_answer_match.group(1).strip()}

        if action_match:
            return {
                "type": "action",
                "thought": thought_match.group(1).strip() if thought_match else "",
                "tool": action_match.group(1).strip(),
                "input": action_match.group(2).strip()
            }

        return {"type": "unknown", "raw": output}
    

    def normalize_tool_name(name):
        return name.lower().replace("_", " ").strip()


    def run_ReAct(tools):
        logger.info(f"Initializing ReAct with max {max_iter} iter.")
        history = ""
        for _ in range(max_iter):
            logger.info(f"ReAct step {_+1}")
            prompt = build_react_prompt(history, tools)
            response = llm.invoke(prompt).content
            #print(f"LLM:\n{response}\n")
            parsed = parse_react_output(response)

            if parsed["type"] == "final":
                return parsed["answer"]

            elif parsed["type"] == "action":
                thought, tool_name, action_input = parsed["thought"], parsed["tool"], parsed["input"]
                logger.info(f"ReAct:\nAction:{tool_name}\nInput:{action_input}")
                norm_tool_name = normalize_tool_name(tool_name)

                tool_key = next((name for name in tools if normalize_tool_name(name) == norm_tool_name), None)
                if not tool_key:
                    history += f"Thought: Tool {tool_name} not found.\nObservation: None\n"
                    continue
                tool = tools[tool_key]

                try:
                    observation = tool["func"](action_input)
                except Exception as e:
                    observation = f"Error: {e}"

                history += f"Thought: \nAction: {tool_name}\nAction Input: {action_input}\nObservation: {observation}\n"
            
            else:
                history += f"Thought: Couldn't parse response.\nObservation: {parsed['raw']}\n"

        return "Could not find final answer within max steps. Please retry."


    tools = set_up_tools()
    return run_ReAct(tools)