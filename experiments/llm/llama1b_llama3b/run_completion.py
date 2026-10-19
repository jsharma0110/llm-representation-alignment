import argparse
import json
import math
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


def load_wikitext():
    print("\nLoading WikiText-2 test set...")

    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    # Combine all non-empty test entries into one continuous stream.
    text = "\n\n".join(
        item
        for item in dataset["text"]
        if item.strip()
    )

    print(f"WikiText entries: {len(dataset)}")

    return text


def calculate_perplexity(
    model,
    tokenizer,
    text,
    device,
    max_length=512,
    stride=256,
    max_windows=0,
):
    """
    Sliding-window perplexity evaluation.

    max_windows=0 means evaluate the full WikiText-2 test set.
    """

    print("\nTokenizing WikiText-2...")

    encodings = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    seq_len = encodings.input_ids.size(1)

    print(f"Total tokens: {seq_len}")
    print(f"Context window used: {max_length}")
    print(f"Stride: {stride}")

    total_nll = 0.0
    total_scored_tokens = 0

    previous_end = 0
    window_count = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)

        # Number of new tokens introduced since the previous window.
        target_length = end - previous_end

        input_ids = encodings.input_ids[:, begin:end].to(device)
        target_ids = input_ids.clone()

        # Ignore tokens already evaluated in the previous overlapping window.
        if target_length < target_ids.size(1):
            target_ids[:, :-target_length] = -100

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                labels=target_ids,
            )

        # Causal LM loss shifts labels internally, so count scored
        # positions after the first token.
        scored_tokens = (
            target_ids[:, 1:] != -100
        ).sum().item()

        if scored_tokens > 0:
            total_nll += outputs.loss.item() * scored_tokens
            total_scored_tokens += scored_tokens

        window_count += 1

        if window_count % 25 == 0:
            running_loss = total_nll / total_scored_tokens
            running_ppl = math.exp(running_loss)

            print(
                f"Window {window_count} | "
                f"running perplexity = {running_ppl:.4f}"
            )

        previous_end = end

        if end == seq_len:
            break

        if max_windows > 0 and window_count >= max_windows:
            break

    average_loss = total_nll / total_scored_tokens
    perplexity = math.exp(average_loss)

    return {
        "perplexity": perplexity,
        "average_loss": average_loss,
        "windows": window_count,
        "scored_tokens": total_scored_tokens,
    }


def evaluate(model_key, max_windows, max_length, stride):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("TEXT COMPLETION / LANGUAGE MODELING")
    print("=" * 70)
    print(f"Device: {device}")

    if device != "cuda":
        raise RuntimeError(
            "CUDA is not available. "
            "Run this experiment inside the Unity GPU allocation."
        )

    model_name = MODELS[model_key]

    text = load_wikitext()
    model, tokenizer = load_model(model_name, device)

    result = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
    )

    result.update(
        {
            "task": "text_completion",
            "dataset": "wikitext-2-raw-v1/test",
            "model_key": model_key,
            "model_name": model_name,
            "max_length": max_length,
            "stride": stride,
            "max_windows_requested": max_windows,
        }
    )

    run_name = (
        "full"
        if max_windows == 0
        else f"windows{max_windows}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        RESULTS_DIR
        / f"completion_{model_key}_{run_name}.json"
    )

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print("Dataset: WikiText-2 test")
    print(f"Perplexity: {result['perplexity']:.4f}")
    print(f"Windows: {result['windows']}")
    print(f"Scored tokens: {result['scored_tokens']}")
    print(f"Saved to: {output_file}")

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
        "--max-windows",
        type=int,
        default=0,
        help="0 = full WikiText test set",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    evaluate(
        model_key=args.model,
        max_windows=args.max_windows,
        max_length=args.max_length,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()