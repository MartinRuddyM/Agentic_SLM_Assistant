from tools.code import run_code
from tools.web_search import web_search
from logger import get_logger
import re
from typing import List

logger = get_logger(__name__)

def ReAct_process(query:str, react_task_desc:str, prompts:List[str], good_llm, cheap_llm, max_iter=10):
    
    def set_up_tools():
        def run_code_wrapper(task_desc):
            return run_code(task_desc, good_llm, prompts)

        def web_search_wrapper(search_terms):
            return web_search(search_terms, query, prompts, cheap_llm)

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
            "task_description":react_task_desc,
            "tool_descriptions":tool_descriptions,
            "tool_names":tool_names,
            "input":query,
            "agent_scratchpad":agent_scratchpad,
        }
        print(prompts["react_step_by_step"].format(**values))
        return prompts["react_step_by_step"].format(**values)


    def parse_react_output(output: str, tool_names: list):
        """
        Parse LLM's output and determine next step based on order:
        - Thought followed by Final Answer => return final answer
        - Thought followed by Action => return action & input

        All names are normalized to avoid possible simple mispelling errors from the LLM

        ALWAYS the LLM output, given the react prompt, should start with a Thought, then
        either an Action with its Action Input OR a Final Answer.
        """

        normalized_tool_names = [re.sub(r'[\W_]+', '', t.lower()) for t in tool_names]
        token_pattern = r"(thought|actioninput|action|observation|finalanswer)"
        matches = re.findall(f"({token_pattern})\s*:\s*(.*?)\s*(?=(?:{token_pattern})\s*:|\Z)", 
                            output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return {"error": "Could not parse any valid Thought/Action/Final Answer entries."}

        steps = []
        for label, content in matches:
            key = re.sub(r'[\W_]+', '', label.lower())
            steps.append((key, content.strip()))
        if not steps or steps[0][0] != "thought":
            return {"error": "First step must be a Thought."}
        thought = steps[0][1]

        if len(steps) > 1:
            next_key, next_content = steps[1]
            if next_key == "finalanswer":
                return {
                    "type": "final_answer",
                    "content": next_content
                }
            elif next_key == "action":
                action_raw = next_content
                normalized_action = re.sub(r'[\W_]+', '', action_raw.lower())
                if normalized_action not in normalized_tool_names:
                    return {"error": f"Unknown action/tool selected: '{action_raw}'."}

                # Try to find corresponding Action Input
                action_input = None
                for k, v in steps[2:]:
                    if k == "actioninput":
                        action_input = v
                        break

                if not action_input:
                    return {"error": f"Action '{action_raw}' given but no Action Input provided."}

                return {
                    "type": "action",
                    "content": {
                        "thought": thought,
                        "action": action_raw,
                        "input": action_input
                    }
                }
        return {"error": "A Thought was parsed, but either there was nothing after it, or nothing useful was identified."}


    logger.info(f"Starting ReAct process with max {max_iter} iterations.")
    tools = set_up_tools()
    agent_scratchpad = ""
    tool_names = list(tools.keys())

    for iteration in range(max_iter):
        logger.info(f"\033[92mReAct step {iteration+1}\033[0m")
        react_prompt = build_react_prompt(agent_scratchpad, tools)
        output = cheap_llm.invoke(react_prompt).content
        parsed = parse_react_output(output, tool_names)

        if parsed.get("error"):
            agent_scratchpad += f"\nObservation: {parsed['error']}\n"
            continue

        if parsed["type"] == "final_answer":
            return parsed["content"]
        
        if parsed["type"] == "action":
            thought = parsed["content"]["thought"]
            action_name = parsed["content"]["action"]
            action_input = parsed["content"]["input"]
            tool_func = tools.get(action_name, {}).get("func")
            if not tool_func:
                agent_scratchpad += f"\nObservation: Tool '{action_name}' not found.\n"
                logger.info(f"ReAct Error: Tool selected not available")
                continue

            try:
                observation = tool_func(action_input)
            except Exception as e:
                observation = f"Tool execution error: {str(e)}"
                logger.info(f"ReAct Error: error in tool execution. Error: {str(e)}")

            agent_scratchpad += (
                f"\nThought: {thought}"
                f"\nAction: {action_name}"
                f"\nAction Input: {action_input}"
                f"\nObservation: {observation}\n"
            )
            

    return "Failed to reach a final answer after maximum iterations."
