import ollama
import time

# Load the model first (this ensures it's ready before querying)
model_name = "deepseek-r1:1.5b"

# Define parameters for efficiency
parameters = {
    "temperature": 0,   # Deterministic output
}

# Measure time taken for inference
start_time = time.time()

test_query = """
Make a two-sentence summary of the text below:

TEXT:

    Hey all, I’m pretty new to the game (I have around 40 hours) and I’m struggling to play Germany. That’s the only nation I’ve played so far, cause I’m trying to specializing on one nation before trying the others, but it seems I can’t win vs AI. I’ve seen plenty 
of tutorials, but everyone seems on a different version of the game cause I have different focus than the one I see on tutorial (as example “four year plan”). I’ve bought the DLC monthly sub, and I think I’m playing on the last version of the game, but a friend of mine which bought the game yesterday (with the monthly sub too) have different menu wallpapers and starting screen/leaders pictures, so idk if that’s a me-problem. Btw, can someone give me some advice on how to play as Germany in single player? I’ve tried to fight France with airtroopers but Poland destroyed me from the back. In the next game I tried to fight Poland with airtroopers, light tank and infrantry but Poland alone sweeped me.

ed so far, cause I’m trying to specializing on one nation before trying the others, but it seems I can’t win vs AI. I’ve seen plenty 
of tutorials, but everyone seems on a different version of the game cause I have different focus than the one I see on tutorial (as example “four year plan”). I’ve bought the DLC monthly sub, and I think I’m playing on the last version of the game, but a friend of mine which bought the game yesterday (with the monthly sub too) have different menu wallpapers and starting screen/leaders pictures, so idk if that’s a me-problem. Btw, can someone give me some advice on how to play as Germany in single player? I’ve tried to fight France with airtroopers but Poland destroyed me from the back. In the next game I tried to fight Poland with airtroopers, light tank and infrantry but Poland alone sweeped me.

xample “four year plan”). I’ve bought the DLC monthly sub, and I think I’m playing on the last version of the game, but a friend of mine which bought the game yesterday (with the monthly sub too) have different menu wallpapers and starting screen/leaders pictures, so idk if that’s a me-problem. Btw, can someone give me some advice on how to play as Germany in single player? I’ve tried to fight France with airtroopers but Poland destroyed me from the back. In the next game I tried to fight Poland with airtroopers, light tank and infrantry but Poland alone sweeped me.

o idk if that’s a me-problem. Btw, can someone give me some advice on how to play as Germany in single player? I’ve tried to fight France with airtroopers but Poland destroyed me from the back. In the next game I tried to fight Poland with airtroopers, light tank and infrantry but Poland alone sweeped me.


d infrantry but Poland alone sweeped me.



              A place to share content, ask questions and/or talk about the grand strategy game Hearts of Iron IV by Paradox Development Studio.
              A place to share content, ask questions and/or talk about the grand strategy game Hearts of Iron IV by Paradox Development Studio.
ent Studio

SUMMARY:
"""

response = ollama.chat(model=model_name, messages=[{"role": "user", "content": test_query}], options=parameters)

end_time = time.time()
elapsed_time = end_time - start_time

print(response["message"]["content"])
print(f"Time taken: {elapsed_time:.2f} seconds")
