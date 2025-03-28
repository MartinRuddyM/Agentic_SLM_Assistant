from tools.llm_functions import *
from typing import List

class Interaction:
    """Stores each Question and Answer pair"""
    
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

    def __repr__(self):
        return f"Interaction(Q: {self.question}, A: {self.answer})"


class Conversation:
    '''Stores each of the interactions.
    Sets up the conversation contextual info.
    Keeps updated summaries.
    On closing, generates the final summary and appends it to DB'''

    def __init__(self, default_chat, cheap_chat, prompts):
        self.default_chat = default_chat
        self.cheap_chat = cheap_chat
        self.prompts = prompts
        self.conversation_history: List[Interaction] = []
        self.recent_summary: str = ""
        self.summary_history: List[str] = []
        self.summary_past_extension = 5 # How many Q&A back to use to create the summary
        print("Type exit to exit the conversation")


    def add_interaction(self, question: str, answer: str):
        self.history.append(Interaction(question, answer))
        self._update_recent_summary()


    def _update_recent_summary(self):
        "Creates a summary of the last N interactions"
        last_n = self.history[-self.summary_past_extension:]
        summary = last_n_interactions_summary(last_n, self.cheap_chat, self.prompts)
        self.summary_history.append(summary)
        self.recent_summary = summary
    

    def _generate_final_summary(self):
        '''Generates final summary of the conversation based on all stored summaries
        It takes partial summaries sequentially to cover all interactions'''
        if len(self.conversation_history) <= self.summary_past_extension:
            return self.recent_summary
        collected_summaries = []
        index = 0
        history_length = len(self.summary_history)
        while index < history_length:
            next_index = min(index + self.summary_past_extension, history_length)
            collected_summaries.append(self.summary_history[next_index - 1])
            index = next_index
        return final_conversation_summary(collected_summaries, self.cheap_chat, self.prompts)
    

    def _retrieve_permanent_user_info(self):
        '''When the conversation ends, looks across all User prompts and decides if there is new
        User information that should be kept. Then stores in DB.'''
        all_user_prompts = "\n\n".join(interaction.question for interaction in self.conversation_history)
        extracted_info = extract_permanent_user_information(all_user_prompts, self.default_chat, self.prompts)
        return extracted_info
        
        

    def exit_conversation(self):
        print("Exiting conversation...")
        if len(self.conversation_history) == 0:
            return None
        return self._generate_final_summary(), self._retrieve_permanent_user_info()

