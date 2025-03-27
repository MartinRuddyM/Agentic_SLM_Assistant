from langchain.agents import Tool
from tools.code import run
from tools.web_search import web_search

from langchain.agents import create_react_agent, AgentExecutor

def ReAct_processing(llm, query:str, prompts, max_iter=10, user_context:str="", debug=False):
    tools = [
        # TO DO: Change the tool description with better prompts
        Tool(name="Web Search", func=web_search, description="Useful for searching the web for current or permanent information"),
        Tool(name="Run Code", func=run, description="Useful for executing Python code"),
    ]
    agent = create_react_agent(llm=llm, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, max_iterations=max_iter, max_execution_time=180, early_stopping_method="generate", handle_parsing_errors=True)
    full_input = f"{user_context}\n\nQuestion: {query}"
    response = agent_executor.invoke({"input":full_input})
    final_answer = response["output"]
    intermediate_steps = "\n".join(f"{action.log}\nObservation: {observation}" for action, observation in response["intermediate_steps"])
    if debug:
        return f"Intermediate steps:\n{intermediate_steps}\n\nFinal answer:\n{final_answer}"
    return final_answer

