from pathlib import Path

import numpy as np
import pandas as pd

from alignment.similarity import (
    compute_procrustes_matrix,
    layer_number,
)


LLAMA1B_DIR = Path("results/llama1b")
LLAMA3B_DIR = Path("results/llama3b")

OUTPUT_DIR = Path("results/llama1b_llama3b")

PCA_DIM = 128


def main():
    matrix, llama1b_files, llama3b_files = (
        compute_procrustes_matrix(
            LLAMA1B_DIR,
            LLAMA3B_DIR,
            pca_dim=PCA_DIM,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_DIR / "procrustes_matrix.npy",
        matrix,
    )

    row_labels = [
        f"llama1b_layer_{layer_number(path)}"
        for path in llama1b_files
    ]

    column_labels = [
        f"llama3b_layer_{layer_number(path)}"
        for path in llama3b_files
    ]

    dataframe = pd.DataFrame(
        matrix,
        index=row_labels,
        columns=column_labels,
    )

    dataframe.to_csv(
        OUTPUT_DIR / "procrustes_matrix.csv"
    )

    best_position = np.unravel_index(
        np.argmax(matrix),
        matrix.shape,
    )

    print(f"Matrix shape: {matrix.shape}")
    print(f"Minimum similarity: {matrix.min():.4f}")
    print(f"Maximum similarity: {matrix.max():.4f}")
    print(
        "Best pair: "
        f"Llama-1B layer {best_position[0]}, "
        f"Llama-3B layer {best_position[1]}"
    )


if __name__ == "__main__":
    main()
