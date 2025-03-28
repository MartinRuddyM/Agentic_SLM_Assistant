from conversation import Interaction
from typing import List
from datetime import datetime

def last_n_interactions_summary(interactions: List[Interaction], llm, prompts):
    """Provides summary of last few interactions within the conversation
    Could be done INCREMENTALLY but might lead to loss of information"""

    system_prompt = prompts["last_n_summary"]
    conversation_text = "\n".join(
        [f"Q: {interaction.question}\nA: {interaction.answer}" for interaction in interactions]
    )
    final_prompt = f"{system_prompt}\n\n{conversation_text}"
    return llm.invoke(final_prompt).content


def final_conversation_summary(summaries: List[str], llm, prompts):
    """Generates a final summary of the entire conversation using stored summaries."""

    system_prompt = prompts["final_summary"]
    summaries_text = "\n".join(summaries)
    final_prompt = f"{system_prompt}\n\nPartial summaries:\n{summaries_text}"
    return llm.invoke(final_prompt).content
    

def extract_permanent_user_information(all_user_prompts: List[str], llm, prompts):
    """Evaluates all user prompts and decides if there is any important information.
    If thre is, it lists and returns it.
    It does not check if the information is already stored in the DB, this should be done separately"""

    system_prompt = prompts["extract_permanent_user_info"]
    all_conversation_user_prompts = "\n".join(all_user_prompts)
    final_prompt = f"{system_prompt}\n\nUser questions:\n{all_conversation_user_prompts}"
    user_info_raw = llm.invoke(final_prompt).content
    user_info = extract_statements(user_info_raw)
    return user_info


def personalize_final_answer(query:str, current_answer: str, user_information: List[str], past_conversations_summaries, conversation, llm, prompts):
    """Given past history of interactions and personal user information,
    adapts the final answer to make reference to past conversations and user info.
    This is to try to give it a more personalized appearance"""
    conversation_summaries = [(summary, datetime.fromisoformat(date).strftime("%d %B")) for summary, date in past_conversations_summaries]
    conversation_summaries = "\n\n".join(f"{date}\n{summary}" for summary, date in conversation_summaries)
    user_info = "\n".join(user_information)
    values = {
        "user_info":user_info,
        "conversation_summaries":conversation_summaries,
        "interactions_summary":conversation.recent_summary,
        "query":query,
        "answer":current_answer
    }
    final_prompt = prompts["personalize_final_asnwer"].format(**values)
    return llm.invoke(final_prompt).content


def extract_statements(text: str) -> list[str]:
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
    
