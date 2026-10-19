import argparse
import csv
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
}

RESULTS_DIR = Path("results/llama1b_llama3b/task_evals")

def load_model(model_name: str, device: str):
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    model.to(device)
    model.eval()

    return model, tokenizer


def load_sst2(limit: int):
    print("\nLoading SST-2 validation set...")

    dataset = load_dataset(
        "glue",
        "sst2",
        split="validation",
    )

    if limit > 0:
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))

    print(f"Examples: {len(dataset)}")

    return dataset


def score_candidate(
    model,
    tokenizer,
    prompt,
    candidate,
    device,
):
    """
    Calculate mean negative log likelihood for one candidate label.

    Lower loss = model assigns greater probability to that label.
    """

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    # The leading space makes the label a natural continuation.
    candidate_ids = tokenizer(
        " " + candidate,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    input_ids = torch.cat(
        [prompt_ids, candidate_ids],
        dim=1,
    )

    labels = torch.full_like(
        input_ids,
        fill_value=-100,
    )

    # Score only the candidate label, not the prompt.
    labels[:, -candidate_ids.size(1):] = candidate_ids

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
        )

    return outputs.loss.item()


def classify(
    model,
    tokenizer,
    sentence,
    device,
):
    prompt = (
        "Classify the sentiment of the following sentence "
        "as positive or negative.\n\n"
        f"Sentence: {sentence}\n\n"
        "Sentiment:"
    )

    positive_loss = score_candidate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate="positive",
        device=device,
    )

    negative_loss = score_candidate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate="negative",
        device=device,
    )

    if positive_loss < negative_loss:
        prediction = 1
        predicted_label = "positive"
    else:
        prediction = 0
        predicted_label = "negative"

    return (
        prediction,
        predicted_label,
        positive_loss,
        negative_loss,
    )


def evaluate(model_key, limit):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("SENTENCE CLASSIFICATION")
    print("=" * 70)
    print(f"Device: {device}")

    if device != "cuda":
        raise RuntimeError(
            "CUDA is not available. "
            "Run this experiment inside the Unity GPU allocation."
        )

    model_name = MODELS[model_key]

    dataset = load_sst2(limit)

    model, tokenizer = load_model(
        model_name,
        device,
    )

    correct = 0
    total = 0

    run_name = (
        "full"
        if limit == 0
        else f"limit{limit}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_file = (
        RESULTS_DIR
        / f"classification_predictions_{model_key}_{run_name}.csv"
    )

    with open(
        prediction_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "sentence",
                "gold_label",
                "predicted_label",
                "correct",
                "positive_loss",
                "negative_loss",
            ],
        )

        writer.writeheader()

        for index, example in enumerate(dataset):
            sentence = example["sentence"]
            gold = int(example["label"])

            (
                prediction,
                predicted_label,
                positive_loss,
                negative_loss,
            ) = classify(
                model=model,
                tokenizer=tokenizer,
                sentence=sentence,
                device=device,
            )

            is_correct = prediction == gold

            if is_correct:
                correct += 1

            total += 1

            gold_label = (
                "positive"
                if gold == 1
                else "negative"
            )

            writer.writerow(
                {
                    "index": index,
                    "sentence": sentence,
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                    "correct": is_correct,
                    "positive_loss": positive_loss,
                    "negative_loss": negative_loss,
                }
            )

            # Print first five examples for sanity checking.
            if index < 5:
                print()
                print(f"Example {index + 1}")
                print(f"Sentence: {sentence}")
                print(f"Gold: {gold_label}")
                print(f"Prediction: {predicted_label}")
                print(
                    f"Positive loss: {positive_loss:.4f}"
                )
                print(
                    f"Negative loss: {negative_loss:.4f}"
                )

            if (index + 1) % 50 == 0:
                running_accuracy = correct / total

                print(
                    f"\n{index + 1}/{len(dataset)} "
                    f"| running accuracy = "
                    f"{running_accuracy * 100:.2f}%"
                )

    accuracy = correct / total

    summary = {
        "task": "sentence_classification",
        "dataset": "glue/sst2/validation",
        "model_key": model_key,
        "model_name": model_name,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "correct": correct,
        "total": total,
        "limit_requested": limit,
    }

    summary_file = (
        RESULTS_DIR
        / f"classification_{model_key}_{run_name}.json"
    )

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print("Dataset: SST-2 validation")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Summary: {summary_file}")
    print(f"Predictions: {prediction_file}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS.keys(),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = full SST-2 validation set",
    )

    args = parser.parse_args()

    evaluate(
        model_key=args.model,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()