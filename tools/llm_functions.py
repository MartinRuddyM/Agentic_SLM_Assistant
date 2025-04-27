from interaction import Interaction
from typing import List
from datetime import datetime
import re
import json
from typing import Counter
import random


def interaction_summary(interaction:Interaction, reasoning, llm, prompts):
    """Provides summary of given interaction within the conversation
    If a reasoning is provided, it tries to get the important
    data also into the summary, so it can be used at later Questions"""

    interaction_text = f"Q: {interaction.question}\n\nA: {interaction.answer}"
    values = {
        "interaction":interaction_text,
    }
    if reasoning:
        interaction_text += f"\n\nReasoning: {reasoning}"
        final_prompt = prompts["interaction_summary_with_reasoning"].format(**values)
    else:
        final_prompt = prompts["interaction_summary"].format(**values)
    return llm.invoke(final_prompt).content


def final_conversation_summary(summaries: List[str], llm, prompts):
    """Generates a final summary of the entire conversation using stored summaries."""

    summaries_text = "\n\n".join(summaries)
    values = {
        "partial_summaries":summaries_text,
    }
    final_prompt = prompts["final_summary"].format(**values)
    return llm.invoke(final_prompt).content
    

def extract_permanent_user_information(all_user_prompts: List[str], llm, prompts):
    """Evaluates all user prompts and decides if there is any important information.
    If thre is, it lists and returns it.
    It does not check if the information is already stored in the DB, this should be done separately"""

    all_conversation_user_prompts = "\n\n".join(all_user_prompts)
    values = {
        "user_questions":all_conversation_user_prompts,
    }
    final_prompt = prompts["extract_permanent_user_info"].format(**values)
    user_info_raw = llm.invoke(final_prompt).content
    user_info = extract_user_statements(user_info_raw)
    print("EXTRACTED USER INFO")
    print(user_info)
    return user_info


def personalize_final_answer(query:str, current_answer: str, user_information: List[str], past_conversations_summaries, conversation, llm, prompts, reference_rate: float=0.2):
    """Given past history of interactions and personal user information,
    adapts the final answer to make reference to past conversations and user info.
    This is to try to give it a more personalized appearance.
    It will include references to previous conversations at a random rate.
    Keeping this rate low ensures the user does not get bombarded with unsolicited suggestions."""

    if len(user_information) == 0 or not past_conversations_summaries:
        return current_answer
    if random.random() < reference_rate:
        conversation_summaries = [(summary, datetime.fromisoformat(date).strftime("%d %B")) for summary, date in past_conversations_summaries]
        #conversation_summaries = "\n\n".join(f"{date}\n{summary}" for summary, date in conversation_summaries)
        #user_info = "\n".join([text for (text,) in user_information])
        values = {
            "user_info":user_information,
            "conversation_summaries":conversation_summaries,
            "interactions_summary":conversation.get_last_n_summaries(n=5),
            "query":query,
            "answer":current_answer
        }
        final_prompt = prompts["personalize_final_asnwer"].format(**values)
    else:
        values = {
            "interactions_summary":conversation.get_last_n_summaries(n=5),
            "query":query,
            "answer":current_answer
        }
        final_prompt = prompts["personalize_final_asnwer_simplified"].format(**values)
    return llm.invoke(final_prompt).content


def contrast_user_information(existing_info: List[str], new_info: List[str], llm, prompts):
    values = {
        "original_statements":"\n".join(existing_info),
        "new_statements":"\n".join(new_info),
    }
    final_prompt = prompts["contrast_user_information"].format(**values)
    user_info_raw = llm.invoke(final_prompt).content
    user_info = extract_user_statements(user_info_raw)
    return user_info


def extract_user_statements(text: str) -> list[str]:
    """Function to extract the LLM-identified statements from the LLM answer.
    It uses Regex with 3 methods in case the LLM did not give a well-structures output.
    It requires an agreement of 2 out of 3 methods. If nothing works, it falls back to the
    last method only. The expected format from the LLM is based on the system prompt for
    extract_permanent_user_information which can be found in prompts.yaml"""
    
    method_results = []

    # --- Method 1: Try to extract JSON block ---
    method1 = set()
    curly_match = re.search(r'\{.*\}', text, re.DOTALL)
    if curly_match:
        try:
            parsed_json = json.loads(curly_match.group())
            if "items" in parsed_json and isinstance(parsed_json["items"], list):
                for item in parsed_json["items"]:
                    cleaned = re.sub(r'^\d+\.\s*', '', item).strip()
                    if cleaned:
                        method1.add(cleaned)
        except json.JSONDecodeError:
            pass
    method_results.append(method1)

    # --- Method 2: Regex to extract "items": [ ... ] block ---
    method2 = set()
    square_match = re.search(r'"items"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if square_match:
        raw_items = square_match.group(1)
        string_matches = re.findall(r'"(.*?)"', raw_items)
        for item in string_matches:
            cleaned = re.sub(r'^\d+\.\s*', '', item).strip()
            if cleaned:
                method2.add(cleaned)
    method_results.append(method2)

    # --- Method 3: Fallback – regex to find all numbered lines anywhere ---
    method3 = set()
    numbered_lines = re.findall(r'\d+\.\s+(.*?)(?=\n|,|"|$)', text)
    for item in numbered_lines:
        cleaned = item.strip().strip('"')
        if cleaned:
            method3.add(cleaned)
    method_results.append(method3)

    # --- Combine and count occurrences across methods ---
    counter = Counter()
    for method in method_results:
        for item in method:
            counter[item] += 1

    # --- Choose reliable entries: at least 2 method votes ---
    reliable_items = [item for item, count in counter.items() if count >= 2]

    # If we have at least 1 reliable item, return them
    if reliable_items:
        return sorted(set(reliable_items))

    # Else, fallback to method 3 only (last method)
    return sorted(method3) if method3 else []
    

def get_react_task_desc(relevant_user_info, past_conversations_summaries, conversation, query:str, llm, prompts):
    conversation_summaries = [(summary, datetime.fromisoformat(date).strftime("%d %B")) for summary, date in past_conversations_summaries]
    conversation_summaries = "\n\n".join(f"{date}\n{summary}" for summary, date in conversation_summaries)
    user_info = "\n".join(text[0] for text in relevant_user_info)
    past_questions = conversation.get_last_n_summaries(n=5)
    values = {
        "query":query,
        "user_info":user_info,
        "previous_messages":past_questions,
        "past_conversations":conversation_summaries,
    }
    final_prompt = prompts["react_get_task_description"].format(**values)
    return llm.invoke(final_prompt).content


def get_react_user_context(relevant_user_info:str, past_conversations_summaries, query:str, llm, prompts):
    conversation_summaries = [(summary, datetime.fromisoformat(date).strftime("%d %B")) for summary, date in past_conversations_summaries]
    conversation_summaries = "\n\n".join(f"{date}\n{summary}" for summary, date in conversation_summaries)
    user_info = "\n".join(text[0] for text in relevant_user_info)
    values = {
        "query": query,
        "past_conversations":conversation_summaries,
        "user_info":user_info,
    }
    final_prompt = prompts["react_get_user_context"].format(**values)
    return llm.invoke(final_prompt).content
