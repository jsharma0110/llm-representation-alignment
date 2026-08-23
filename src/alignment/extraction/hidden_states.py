from pathlib import Path

import torch

from alignment.extraction.pooling import pool_hidden_state


def extract_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    output_dir: str | Path,
    pooling: str = "mean",
    max_length: int = 512,
    log_every: int = 10,
) -> None:
    """
    Extract and save pooled hidden states from every model layer.

    Args:
        model:
            Hugging Face causal language model.

        tokenizer:
            Tokenizer corresponding to the model.

        texts:
            Input strings used for representation extraction.

        output_dir:
            Directory where layer tensors are saved.

        pooling:
            Token pooling strategy.

        max_length:
            Maximum tokenized sequence length.

        log_every:
            Print progress after this many examples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_outputs = None

    for idx, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states

        if layer_outputs is None:
            layer_outputs = [
                [] for _ in range(len(hidden_states))
            ]

        for layer_idx, layer in enumerate(hidden_states):
            embedding = pool_hidden_state(
                layer,
                strategy=pooling,
            ).squeeze(0).cpu()

            layer_outputs[layer_idx].append(embedding)

        if (idx + 1) % log_every == 0:
            print(
                f"Processed {idx + 1}/{len(texts)} examples"
            )

    if layer_outputs is None:
        raise ValueError("No input texts were provided.")

    print("Saving tensors...")

    for layer_idx, embeddings in enumerate(layer_outputs):
        tensor = torch.stack(embeddings)

        torch.save(
            tensor,
            output_dir / f"layer_{layer_idx}.pt",
        )

    print(f"Saved hidden states to {output_dir}")
