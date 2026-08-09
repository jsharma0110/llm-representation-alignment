from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alignment.similarity import (
    fit_pca,
    heldout_similarity,
    layer_number,
    load_tensor,
)
from alignment.similarity.procrustes import (
    get_layer_files,
)


LLAMA1B_DIR = Path("results/hidden_states/llama1b")
LLAMA3B_DIR = Path("results/hidden_states/llama3b")

OUTPUT_DIR = Path("results/llama1b_llama3b")

PCA_DIM = 64
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42


def main():
    torch.manual_seed(RANDOM_SEED)

    llama1b_files = get_layer_files(
        LLAMA1B_DIR
    )

    llama3b_files = get_layer_files(
        LLAMA3B_DIR
    )

    number_of_samples = load_tensor(
        llama1b_files[0]
    ).shape[0]

    if (
        load_tensor(llama3b_files[0]).shape[0]
        != number_of_samples
    ):
        raise ValueError(
            "The two model result folders contain "
            "different numbers of samples."
        )

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    indices = torch.randperm(
        number_of_samples,
        generator=generator,
    )

    train_size = int(
        number_of_samples * TRAIN_FRACTION
    )

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    shuffled_order = torch.randperm(
        len(test_indices),
        generator=generator,
    )

    shuffled_test_indices = test_indices[
        shuffled_order
    ]

    llama1b_reduced = []
    llama3b_reduced = []

    for path in llama1b_files:
        representations = load_tensor(path)

        train, test, _, _ = fit_pca(
            representations[train_indices],
            representations[test_indices],
            PCA_DIM,
        )

        llama1b_reduced.append(
            (train, test)
        )

    for path in llama3b_files:
        representations = load_tensor(path)

        (
            train,
            test,
            train_mean,
            directions,
        ) = fit_pca(
            representations[train_indices],
            representations[test_indices],
            PCA_DIM,
        )

        shuffled_test = (
            representations[shuffled_test_indices]
            - train_mean
        ) @ directions

        llama3b_reduced.append(
            (
                train,
                test,
                shuffled_test,
            )
        )

    matrix = np.zeros(
        (
            len(llama1b_files),
            len(llama3b_files),
        ),
        dtype=np.float32,
    )

    shuffled_matrix = np.zeros_like(
        matrix
    )

    for i, (
        source_train,
        source_test,
    ) in enumerate(llama1b_reduced):
        for j, (
            target_train,
            target_test,
            shuffled_target_test,
        ) in enumerate(llama3b_reduced):

            matrix[i, j] = heldout_similarity(
                source_train,
                source_test,
                target_train,
                target_test,
            )

            shuffled_matrix[i, j] = (
                heldout_similarity(
                    source_train,
                    source_test,
                    target_train,
                    shuffled_target_test,
                )
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_DIR / "procrustes_heldout.npy",
        matrix,
    )

    np.save(
        OUTPUT_DIR / "procrustes_heldout_shuffled.npy",
        shuffled_matrix,
    )

    row_names = [
        f"llama1b_layer_{layer_number(path)}"
        for path in llama1b_files
    ]

    column_names = [
        f"llama3b_layer_{layer_number(path)}"
        for path in llama3b_files
    ]

    pd.DataFrame(
        matrix,
        index=row_names,
        columns=column_names,
    ).to_csv(
        OUTPUT_DIR / "procrustes_heldout.csv"
    )

    print(f"Matrix shape: {matrix.shape}")
    print(
        f"Mean real similarity: "
        f"{matrix.mean():.6f}"
    )
    print(
        f"Mean shuffled similarity: "
        f"{shuffled_matrix.mean():.6f}"
    )
    print(
        f"Mean real-minus-shuffled gap: "
        f"{(matrix - shuffled_matrix).mean():.6f}"
    )


if __name__ == "__main__":
    main()
