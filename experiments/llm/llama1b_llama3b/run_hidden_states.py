from alignment.datasets import load_truthfulqa_questions
from alignment.extraction import extract_hidden_states
from alignment.models import load_causal_lm


MODELS = {
    "llama1b": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "output_dir": "results/hidden_states/llama1b",
    },
    "llama3b": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "output_dir": "results/hidden_states/llama3b",
    },
}

NUM_QUESTIONS = 500


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
