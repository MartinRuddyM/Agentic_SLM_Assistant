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
    """Stores each of the interactions.
    Sets up the conversation contextual info.
    Keeps updated summaries.
    On closing, generates the final summary of the full conversation."""

    def __init__(self, default_chat, cheap_chat, prompts):
        self.default_chat = default_chat
        self.cheap_chat = cheap_chat
        self.prompts = prompts
        self.history: List[Interaction] = []
        self.summaries: List[str] = []
        print("Type exit to exit the conversation")


    def add_interaction(self, question: str, answer: str):
        "Add new Interaction to the Conversation"
        new_interaction = Interaction(question, answer)
        self.history.append(new_interaction)
        self._create_interaction_summary(new_interaction)


    def _create_interaction_summary(self, interaction):
        "Creates a summary of the given interaction"
        summary = interaction_summary(interaction, self.cheap_chat, self.prompts)
        self.summaries.append(summary)
    

    def _generate_final_summary(self):
        '''Generates and returns a final summary of the conversation based on all stored summaries.
        Basically it is a summary made of all the partial summaries'''
        final_summary = final_conversation_summary(self.summaries, self.cheap_chat, self.prompts)
        return final_summary
    

    def _retrieve_permanent_user_info(self):
        '''When the conversation ends, looks across all User prompts and decides if there is new
        User information that should be kept. Then returns it to be stored in DB.'''
        all_user_prompts = "\n\n".join(interaction.question for interaction in self.history)
        extracted_info = extract_permanent_user_information(all_user_prompts, self.default_chat, self.prompts)
        return extracted_info
        

    def exit_conversation(self):
        print("Exiting conversation...")
        if len(self.history) == 0:
            return None
        return self._generate_final_summary(), self._retrieve_permanent_user_info()
    

    def get_last_n_summaries(self, n=5):
        """Returns the last n available summaries, or m<n if only m available.
        If none is available, returns empty string"""
        if len(self.summaries) == 0:
            return ""
        limit = min(n, len(self.summaries))
        return "\n\n".join(self.summaries[-limit:])

