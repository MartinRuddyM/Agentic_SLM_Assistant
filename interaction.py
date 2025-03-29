class Interaction:
    """Stores each Question and Answer pair"""
    
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

    def __repr__(self):
        return f"Interaction(Q: {self.question}, A: {self.answer})"