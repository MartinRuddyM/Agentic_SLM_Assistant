from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from tools.code import run_code
from tools.web_search import web_search
from logger import get_logger

logger = get_logger(__name__)


def ReAct_process(query:str, prompts, llm, summarizer_llm, max_iter=10, debug=False):
    def run_code_wrapper(task_desc):
        return run_code(task_desc, llm, prompts)
    
    def web_search_wrapper(search_terms):
        return web_search(search_terms, query, prompts, summarizer_llm)
    
    tools = [
        Tool(name="Web Search", func=web_search_wrapper, description=prompts["web_search_tool_description"]),
        Tool(name="Run Code", func=run_code_wrapper, description=prompts["code_tool_description"]),
    ]
    #####################
    ############TODO
    #####################
    # Adjust react prompt so it makes sense with my custom system prompt and I dont have to duplicate
    prompt = PromptTemplate.from_template(prompts["react_prompt"])
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt, stop_sequence=["Final Answer", "Final answer", "finals answer"])
    logger.info("Starting ReAct process to process the query")
    agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=False, max_iterations=max_iter, max_execution_time=180, early_stopping_method="generate", handle_parsing_errors=True, callbacks=[ReActLoggingHandler()])
    response = agent_executor.invoke({"input":query})
    logger.info("ReAct processing finished")
    print(response)
    final_answer = response["output"]
    #intermediate_steps = "\n".join(f"{action.log}\nObservation: {observation}" for action, observation in response["intermediate_steps"])
    #if debug:
        #return intermediate_steps, final_answer
    return None, final_answer


class ReActLoggingHandler(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        logger.info(f"[Agent Action] {action.log}")
    
    def on_tool_start(self, tool, input_str, **kwargs):
        logger.info(f"[Tool Start] {tool}({input_str})")
    
    def on_tool_end(self, output, **kwargs):
        logger.info(f"[Tool End] => {output}")
    
    def on_agent_finish(self, finish, **kwargs):
        logger.info(f"[Agent Finish] {finish.log}")