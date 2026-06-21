from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True
)

question = "Where did fortune cookies originate?"

inputs = tokenizer(
    question,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.hidden_states
print("Number of layers:", len(hidden_states))
print("First layer shape:", hidden_states[0].shape)
print("Last layer shape:", hidden_states[-1].shape)