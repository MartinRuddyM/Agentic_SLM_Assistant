from conversation import Interaction
from typing import List

def last_n_interactions_summary(interactions: List[Interaction], llm, prompts):
    """Provides summary of last few interactions within the conversation
    Could be done INCREMENTALLY but might lead to loss of information"""

    system_prompt = prompts["last_n_summary"]
    conversation_text = "\n".join(
        [f"Q: {interaction.question}\nA: {interaction.answer}" for interaction in interactions]
    )
    final_prompt = f"{system_prompt}\n\n{conversation_text}"
    return llm.predict(final_prompt).strip()

def final_conversation_summary(summaries: List[str], llm, prompts):
    """Generates a final summary of the entire conversation using stored summaries."""

    system_prompt = prompts["final_summary"]
    summaries_text = "\n".join(summaries)
    final_prompt = f"{system_prompt}\n\nPartial summaries:\n{summaries_text}"
    return llm.predict(final_prompt).strip()
    
def extract_permanent_user_information(all_user_prompts: List[str], llm, prompts):
    """Evaluates all user prompts and decides if there is any important information.
    If thre is, it lists and returns it.
    It does not check if the information is already stored in the DB, this should be done separately"""

    system_prompt = prompts["extract_permanent_user_info"]
    all_conversation_user_prompts = "\n".join(all_user_prompts)
    final_prompt = f"{system_prompt}\n\nUser questions:\n{all_conversation_user_prompts}"
    return llm.predict(final_prompt).strip()

def personalize_final_answer(current_answer: str, user_info: List[str], past_conversations, interactions: List[Interaction], llm, prompts):
    """Given past history of interactions and personal user information,
    adapts the final answer to make reference to past conversations and user info.
    This is to try to give it a more personalized appearance"""
    system_prompt = prompts["personalize_final_asnwer"]
    
    
