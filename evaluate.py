import json
from ollama import chat

with open("factual_qa.json") as f:
    dataset = json.load(f)

models = [
    "tinyllama",
    "qwen2.5:3b"
]

for model in models:

    correct = 0

    print(f"\nTesting {model}")

    for sample in dataset:

        response = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": sample["question"]
                }
            ]
        )

        prediction = response["message"]["content"]

        print("\nQuestion:", sample["question"])
        print("Prediction:", prediction)

        if sample["answer"].lower() in prediction.lower():
            correct += 1

    accuracy = correct / len(dataset)

    print("\nAccuracy:", accuracy)