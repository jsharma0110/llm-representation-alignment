from pathlib import Path

import numpy as np

from alignment.visualization import (
    normalized_diagonal_score,
    plot_heldout_comparison,
)


REAL_MATRIX_PATH = Path(
    "results/llama1b_llama3b/procrustes_heldout.npy"
)

SHUFFLED_MATRIX_PATH = Path(
    "results/llama1b_llama3b/procrustes_heldout_shuffled.npy"
)

OUTPUT_PNG = Path(
    "figures/llama1b_llama3b/procrustes_heldout.png"
)

OUTPUT_PDF = Path(
    "figures/llama1b_llama3b/procrustes_heldout.pdf"
)


def main():
    real_matrix = np.load(
        REAL_MATRIX_PATH
    )

    shuffled_matrix = np.load(
        SHUFFLED_MATRIX_PATH
    )

    plot_heldout_comparison(
        real_matrix=real_matrix,
        shuffled_matrix=shuffled_matrix,
        output_png=OUTPUT_PNG,
        output_pdf=OUTPUT_PDF,
        title=(
            "Llama-3.2-1B vs Llama-3.2-3B "
            "Held-out Procrustes Analysis"
        ),
        x_label="Llama-3.2-3B layer",
        y_label="Llama-3.2-1B layer",
    )

    (
        real_diagonal,
        real_off_diagonal,
        real_advantage,
    ) = normalized_diagonal_score(
        real_matrix
    )

    (
        shuffled_diagonal,
        shuffled_off_diagonal,
        shuffled_advantage,
    ) = normalized_diagonal_score(
        shuffled_matrix
    )

    best_position = np.unravel_index(
        np.argmax(real_matrix),
        real_matrix.shape,
    )

    print(
        f"Matrix shape: "
        f"{real_matrix.shape}"
    )

    print(
        f"Best real layer pair: "
        f"Llama-1B layer {best_position[0]} and "
        f"Llama-3B layer {best_position[1]}"
    )

    print("\nReal matrix:")

    print(
        f"  Near-diagonal mean: "
        f"{real_diagonal:.6f}"
    )

    print(
        f"  Off-diagonal mean: "
        f"{real_off_diagonal:.6f}"
    )

    print(
        f"  Diagonal advantage: "
        f"{real_advantage:.6f}"
    )

    print("\nShuffled matrix:")

    print(
        f"  Near-diagonal mean: "
        f"{shuffled_diagonal:.6f}"
    )

    print(
        f"  Off-diagonal mean: "
        f"{shuffled_off_diagonal:.6f}"
    )

    print(
        f"  Diagonal advantage: "
        f"{shuffled_advantage:.6f}"
    )

    print(
        f"\nSaved PNG to: "
        f"{OUTPUT_PNG}"
    )

    print(
        f"Saved PDF to: "
        f"{OUTPUT_PDF}"
    )


if __name__ == "__main__":
    main()
