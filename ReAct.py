from tools.code import run_code
from tools.web_search import web_search
from logger import get_logger
import re
from typing import List

logger = get_logger(__name__)

def ReAct_process(query:str, basic_user_context:str, prompts:List[str], good_llm, cheap_llm, max_iter=5):
    
    def set_up_tools():
        def run_code_wrapper(task_desc):
            return run_code(task_desc, good_llm, prompts)

        def web_search_wrapper(search_terms):
            return web_search(search_terms, query, prompts, cheap_llm)

        tools = {
            "Web Search": {
                "func": web_search_wrapper,
                "description": prompts["web_search_tool_description"],
                "normalized_name": re.sub(r'[\W_]+', '', "Web Search".lower())
            },
            "Run Code": {
                "func": run_code_wrapper,
                "description": prompts["code_tool_description"],
                "normalized_name": re.sub(r'[\W_]+', '', "Run Code".lower())
            }
        }
        return tools

    
    def build_react_prompt(agent_scratchpad, tools):
        tool_descriptions = "\n".join(
            f"{name}: {tool['description']}" for name, tool in tools.items()
        )
        tool_names = ", ".join(tools.keys())
        values = {
            "user_context":basic_user_context,
            "tool_descriptions":tool_descriptions,
            "tool_names":tool_names,
            "input":query,
            "agent_scratchpad":agent_scratchpad,
        }
        final_prompt = prompts["react_step_by_step"].format(**values)
        print("\033[94m" + final_prompt + "\033[0m")
        return final_prompt


    def parser(output:str):
        token_pattern = (
            r"question"
            r"|thought"
            r"|action[\s_]*input"
            r"|action"
            r"|observation"
            r"|final[\s_]*answer"
        )

        pattern = rf"(?i)\b({token_pattern})\b\s*:\s*(.*?)\s*(?=\b(?:{token_pattern})\b\s*:|\Z)"
        matches = re.findall(pattern, output, re.DOTALL)

        if not matches:
            return []

        steps = []
        for label, content in matches:
            normalized_label = re.sub(r'[\W_]+', '', label.lower())  # normalize e.g. Action Input → actioninput
            if normalized_label != "question":
                steps.append((normalized_label, content.strip()))
        return steps


    def decision_logic(steps, tool_names, taken_decisions):
        normalized_tool_names = [re.sub(r'[\W_]+', '', t.lower()) for t in tool_names]
        for i, (label, content) in enumerate(steps):
            if label == "action":
                tool_call = re.sub(r'[\W_]+', '', content.lower())
                if tool_call not in normalized_tool_names:
                    continue
                # Avoid calling same tool 3 times in a row
                if len(taken_decisions) >= 2 and all(t[1] == tool_call for t in taken_decisions[-2:]):
                    continue

                if i + 1 < len(steps) and steps[i + 1][0] == "actioninput":
                    input_value = steps[i + 1][1]
                    return {
                        "type": "tool_call",
                        "tool": tool_call,
                        "input": input_value,
                        "agent_scratchpad": steps[:i+1]
                    }
                else:
                    return {
                        "type": "error",
                        "content": f"Expected 'Action Input' immediately after action '{content}', but not found.",
                    }

        # No valid action, look for final answer
        for label, content in steps:
            if label == "finalanswer":
                return {
                    "type": "final_answer",
                    "content": content,
                }

        return {
            "type": "error",
            "content": 'No valid action or final answer found in the steps. Make sure your steps follow the available tools or provide a clear "Final answer"',
        }
    

    logger.info(f"Starting ReAct process with max {max_iter} iterations.")
    tools = set_up_tools()
    agent_scratchpad = ""
    tool_names = list(tools.keys())
    taken_decisions = []

    for iteration in range(max_iter):
        logger.info(f"\033[92mReAct step {iteration+1}\033[0m")
        react_prompt = build_react_prompt(agent_scratchpad, tools)
        output = good_llm.invoke(react_prompt).content
        print("\033[31m" +  "MODEL OUTPUT\n" + output + "\033[0m")
        parsed_steps = parser(output)
        decision = decision_logic(parsed_steps, tool_names, taken_decisions)
        if decision["type"] == "error":
            taken_decisions.append(("error", None))
            agent_scratchpad += decision["content"]
        elif decision["type"] == "final_answer":
            return decision["content"], agent_scratchpad
        elif decision["type"] == "tool_call" and iteration + 1 < max_iter:
            tool_name_normalized = decision["tool"]
            taken_decisions.append(("action", tool_name_normalized))
            input_value = decision["input"]
            tool = next((t for t in tools.values() if t["normalized_name"] == tool_name_normalized), None)
            if not tool:
                agent_scratchpad += f"Error: Tool '{tool_name_normalized}' not found.\n"
                continue
            result = tool["func"](input_value) # Call the tool
            # Ahora mismo no utiliza la info de agent scratchpad que pueda devolver el parser
            # (el modelo a veces inventa steps adicionales de razonamiento)
            # Solo se anade actualmente la llamada a la tool y su resultado
            agent_scratchpad += (
                f"Action: {tool_name_normalized}\n"
                f"Action Input: {input_value}\n"
                f"Observation: {result}\n"
            )

    return "Failed to reach a final answer after maximum iterations.", None
                







    def parse_react_output(output: str, tool_names: list):
        """
        Parse LLM's output and determine next step based on order:
        - Thought followed by Final Answer => return final answer
        - Thought followed by Action => return action & input

        All names are normalized to avoid possible simple mispelling errors from the LLM

        ALWAYS the LLM output, given the react prompt, should start with a Thought, then
        either an Action with its Action Input OR a Final Answer.
        """

        # Normalize tool names for robust matching
        normalized_tool_names = [re.sub(r'[\W_]+', '', t.lower()) for t in tool_names]

        # Flexible token pattern: accepts spaces, underscores, etc.
        token_pattern = (
            r"question"
            r"|thought"
            r"|action[\s_]*input"
            r"|action"
            r"|observation"
            r"|final[\s_]*answer"
        )

        # Extract labeled blocks
        pattern = rf"(?i)\b({token_pattern})\b\s*:\s*(.*?)\s*(?=\b(?:{token_pattern})\b\s*:|\Z)"
        matches = re.findall(pattern, output, re.DOTALL)

        if not matches:
            return {"error": "Could not parse any valid steps (Thought, Action, Final Answer, etc.)."}

        # Build normalized (label, content) pairs
        steps = []
        for label, content in matches:
            key = re.sub(r'[\W_]+', '', label.lower())  # normalize label (e.g., Action Input → actioninput)
            steps.append((key, content.strip()))

        # Remove 'question' entries
        steps = [(k, v) for k, v in steps if k != "question"]

        if not steps:
            return {"error": "No valid steps found after removing 'Question'."}

        last_thought = None

        for i, (key, value) in enumerate(steps):
            if key == "thought":
                last_thought = value

            elif key == "finalanswer":
                return {
                    "type": "final_answer",
                    "content": value
                }

            elif key == "action":
                action_raw = value
                normalized_action = re.sub(r'[\W_]+', '', action_raw.lower())

                if normalized_action not in normalized_tool_names:
                    return {"error": f"Unknown action/tool selected: '{action_raw}'."}

                # Look ahead for Action Input
                action_input = None
                for j in range(i + 1, len(steps)):
                    next_key, next_val = steps[j]
                    if next_key == "actioninput":
                        action_input = next_val
                        break
                    elif next_key in ["thought", "action", "finalanswer", "observation"]:
                        break  # Stop if another main step appears before input

                if not action_input:
                    return {"error": f"Action '{action_raw}' given but no Action Input provided."}

                if not last_thought:
                    return {"error": "Action found, but no preceding Thought."}

                return {
                    "type": "action",
                    "content": {
                        "thought": last_thought,
                        "action": action_raw,
                        "input": action_input
                    }
                }

        return {"error": "No actionable step found (Final Answer or Action missing)."}


    logger.info(f"Starting ReAct process with max {max_iter} iterations.")
    tools = set_up_tools()
    agent_scratchpad = ""
    tool_names = list(tools.keys())

    for iteration in range(max_iter):
        logger.info(f"\033[92mReAct step {iteration+1}\033[0m")
        react_prompt = build_react_prompt(agent_scratchpad, tools)
        output = cheap_llm.invoke(react_prompt).content
        parsed = parse_react_output(output, tool_names)
        print(f"\n\nOUTPUT:\n{output}\n\nPARSED\n\n{parsed}\n\n")

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
