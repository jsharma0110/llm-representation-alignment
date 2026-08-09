from alignment.datasets import load_truthfulqa_questions
from alignment.extraction import extract_hidden_states
from alignment.models import load_causal_lm


MODELS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "output_dir": "results/hidden_states/tinyllama",
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "output_dir": "results/hidden_states/qwen",
    },
}

NUM_QUESTIONS = 100


def main():
    questions = load_truthfulqa_questions(
        limit=NUM_QUESTIONS,
    )

    for model_label, config in MODELS.items():
        print(f"\n=== {model_label} ===")

        model, tokenizer = load_causal_lm(
            config["name"]
        )

        extract_hidden_states(
            model=model,
            tokenizer=tokenizer,
            texts=questions,
            output_dir=config["output_dir"],
            pooling="mean",
        )


if __name__ == "__main__":
    main()
