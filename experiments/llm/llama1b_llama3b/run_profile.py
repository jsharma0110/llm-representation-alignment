import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
}

RESULTS_DIR = Path(
    "results/llama1b_llama3b/profiling"
)

PROMPT = (
    "Representation alignment studies whether neural networks "
    "learn similar internal structures and whether these "
    "representations can be transferred between models. "
)


def synchronize():
    """Wait for all queued CUDA operations to finish."""
    torch.cuda.synchronize()


def load_model_and_tokenizer(
    model_name: str,
    device: str,
):
    """
    Measure model loading time.

    Loading time includes:
      1. AutoModelForCausalLM.from_pretrained(...)
      2. moving the model onto the GPU

    Tokenizer loading is performed BEFORE timing and is not
    included in model load time.
    """

    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        "Loading model and moving it to GPU..."
    )

    start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )

    model.to(device)
    model.eval()

    synchronize()

    end = time.perf_counter()

    load_time_seconds = end - start

    return model, tokenizer, load_time_seconds


def prepare_input(
    tokenizer,
    device: str,
    seq_length: int,
):
    """
    Build one fixed input of exactly seq_length tokens.

    The same input length is used for every model so forward
    latency is directly comparable.
    """

    # Repeat the fixed prompt enough times to exceed the desired
    # sequence length, then truncate/pad to exactly seq_length.
    text = PROMPT * 20

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_length,
        padding="max_length",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded[
        "attention_mask"
    ].to(device)

    print(
        f"Input shape: {tuple(input_ids.shape)}"
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def time_first_forward(
    model,
    inputs,
):
    """
    Measure the first forward pass after model loading.

    No warmup is performed before this measurement.
    """

    synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        model(
            **inputs,
            use_cache=False,
        )

    synchronize()

    end = time.perf_counter()

    return (end - start) * 1000.0


def warm_up(
    model,
    inputs,
    warmup_runs: int,
):
    print(
        f"\nRunning {warmup_runs} warmup forwards..."
    )

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(
                **inputs,
                use_cache=False,
            )

    synchronize()


def benchmark_forward(
    model,
    inputs,
    runs: int,
):
    """
    Measure model.forward() latency over repeated runs.

    Each measurement is synchronized so asynchronous CUDA
    execution does not corrupt the timing.
    """

    print(
        f"Running {runs} measured forwards..."
    )

    latencies_ms = []

    with torch.inference_mode():
        for i in range(runs):

            synchronize()

            start = time.perf_counter()

            model(
                **inputs,
                use_cache=False,
            )

            synchronize()

            end = time.perf_counter()

            latency_ms = (
                end - start
            ) * 1000.0

            latencies_ms.append(
                latency_ms
            )

            if (i + 1) % 10 == 0:
                print(
                    f"{i + 1}/{runs} completed"
                )

    return latencies_ms


def evaluate_model(
    model_key: str,
    runs: int,
    warmup_runs: int,
    seq_length: int,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this script "
            "inside a Unity GPU allocation."
        )

    device = "cuda"

    model_name = MODELS[model_key]

    gpu_name = torch.cuda.get_device_name(0)

    gpu_memory_gb = (
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / (1024 ** 3)
    )

    print("\n" + "=" * 72)
    print("LLM GPU PROFILING")
    print("=" * 72)

    print(f"Model: {model_name}")
    print(f"GPU: {gpu_name}")
    print(
        f"GPU memory: "
        f"{gpu_memory_gb:.1f} GB"
    )
    print(f"Precision: float16")
    print(f"Batch size: 1")
    print(
        f"Sequence length: {seq_length}"
    )
    print(f"Measured runs: {runs}")

    (
        model,
        tokenizer,
        load_time_seconds,
    ) = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
    )

    inputs = prepare_input(
        tokenizer=tokenizer,
        device=device,
        seq_length=seq_length,
    )

    print("\nMeasuring first forward...")

    first_forward_ms = (
        time_first_forward(
            model=model,
            inputs=inputs,
        )
    )

    warm_up(
        model=model,
        inputs=inputs,
        warmup_runs=warmup_runs,
    )

    latencies_ms = benchmark_forward(
        model=model,
        inputs=inputs,
        runs=runs,
    )

    mean_latency_ms = statistics.mean(
        latencies_ms
    )

    median_latency_ms = statistics.median(
        latencies_ms
    )

    min_latency_ms = min(
        latencies_ms
    )

    max_latency_ms = max(
        latencies_ms
    )

    result = {
        "model_key": model_key,
        "model_name": model_name,
        "gpu": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "torch_version": torch.__version__,
        "precision": "float16",
        "batch_size": 1,
        "sequence_length": seq_length,
        "warmup_runs": warmup_runs,
        "measured_runs": runs,
        "model_load_time_seconds":
            load_time_seconds,
        "first_forward_ms":
            first_forward_ms,
        "mean_forward_ms":
            mean_latency_ms,
        "median_forward_ms":
            median_latency_ms,
        "min_forward_ms":
            min_latency_ms,
        "max_forward_ms":
            max_latency_ms,
        "all_forward_latencies_ms":
            latencies_ms,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        RESULTS_DIR
        / f"profile_{model_key}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("FINAL PROFILING RESULT")
    print("=" * 72)

    print(
        f"Model load time: "
        f"{load_time_seconds:.4f} s"
    )

    print(
        f"First forward: "
        f"{first_forward_ms:.4f} ms"
    )

    print(
        f"Mean forward: "
        f"{mean_latency_ms:.4f} ms"
    )

    print(
        f"Median forward: "
        f"{median_latency_ms:.4f} ms"
    )

    print(
        f"Min forward: "
        f"{min_latency_ms:.4f} ms"
    )

    print(
        f"Max forward: "
        f"{max_latency_ms:.4f} ms"
    )

    print(
        f"Saved to: {output_file}"
    )

    del model

    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "llama1b",
            "llama3b",
            "all",
        ],
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help=(
            "Number of measured forward "
            "passes."
        ),
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    if args.model == "all":
        model_keys = [
            "llama1b",
            "llama3b",
        ]
    else:
        model_keys = [
            args.model
        ]

    for model_key in model_keys:
        evaluate_model(
            model_key=model_key,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            seq_length=args.seq_length,
        )


if __name__ == "__main__":
    main()