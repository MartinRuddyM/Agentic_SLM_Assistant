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

response = ollama.chat(model=model_name, messages=[{"role": "user", "content": "Where is Rovaniemi?"}], options=parameters)

end_time = time.time()
elapsed_time = end_time - start_time

print(response["message"]["content"])
print(f"Time taken: {elapsed_time:.2f} seconds")
