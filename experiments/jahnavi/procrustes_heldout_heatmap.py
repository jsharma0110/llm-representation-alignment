from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REAL_MATRIX_PATH = (
    PROJECT_ROOT
    / "results"
    / "procrustes_heldout_llama1b_llama3b.npy"
)

SHUFFLED_MATRIX_PATH = (
    PROJECT_ROOT
    / "results"
    / "procrustes_heldout_shuffled_llama1b_llama3b.npy"
)

OUTPUT_PNG = (
    PROJECT_ROOT
    / "figures"
    / "procrustes_heldout_llama1b_llama3b.png"
)

OUTPUT_PDF = (
    PROJECT_ROOT
    / "figures"
    / "procrustes_heldout_llama1b_llama3b.pdf"
)


def add_heatmap(
    axis,
    matrix: np.ndarray,
    title: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
):
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    axis.set_title(title)
    axis.set_xlabel("Llama-3.2-3B layer")
    axis.set_ylabel("Llama-3.2-1B layer")

    axis.set_xticks(
        np.arange(matrix.shape[1])
    )
    axis.set_xticklabels(
        np.arange(matrix.shape[1]),
        rotation=90,
        fontsize=7,
    )

    axis.set_yticks(
        np.arange(matrix.shape[0])
    )
    axis.set_yticklabels(
        np.arange(matrix.shape[0]),
        fontsize=8,
    )

    colorbar = plt.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(colorbar_label)

    return image


def normalized_diagonal_score(
    matrix: np.ndarray,
    width: float = 0.1,
) -> tuple[float, float, float]:
    """
    Compare values near the normalized-depth diagonal against
    values away from it.

    Since the models have different numbers of layers, layer
    positions are converted to values between 0 and 1.
    """
    number_of_rows, number_of_columns = matrix.shape

    row_depths = np.linspace(
        0.0,
        1.0,
        number_of_rows,
    )

    column_depths = np.linspace(
        0.0,
        1.0,
        number_of_columns,
    )

    distance_from_diagonal = np.abs(
        row_depths[:, None] - column_depths[None, :]
    )

    diagonal_mask = distance_from_diagonal <= width
    off_diagonal_mask = distance_from_diagonal > width

    diagonal_mean = float(
        matrix[diagonal_mask].mean()
    )

    off_diagonal_mean = float(
        matrix[off_diagonal_mask].mean()
    )

    diagonal_advantage = (
        diagonal_mean - off_diagonal_mean
    )

    return (
        diagonal_mean,
        off_diagonal_mean,
        diagonal_advantage,
    )


def main() -> None:
    real_matrix = np.load(
        REAL_MATRIX_PATH
    )

    shuffled_matrix = np.load(
        SHUFFLED_MATRIX_PATH
    )

    if real_matrix.shape != shuffled_matrix.shape:
        raise ValueError(
            "The real and shuffled matrices have "
            "different shapes."
        )

    difference_matrix = (
        real_matrix - shuffled_matrix
    )

    OUTPUT_PNG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Percentile scaling makes differences in the highly
    # saturated real matrix easier to see.
    real_vmin = float(
        np.percentile(real_matrix, 5)
    )
    real_vmax = float(
        np.percentile(real_matrix, 99)
    )

    shuffled_limit = float(
        np.max(np.abs(shuffled_matrix))
    )

    difference_vmin = float(
        np.percentile(difference_matrix, 1)
    )
    difference_vmax = float(
        np.percentile(difference_matrix, 99)
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(24, 7),
    )

    add_heatmap(
        axis=axes[0],
        matrix=real_matrix,
        title=(
            "Held-out Procrustes Similarity\n"
            "Real Prompt Correspondence"
        ),
        colorbar_label="Held-out cosine similarity",
        vmin=real_vmin,
        vmax=real_vmax,
    )

    add_heatmap(
        axis=axes[1],
        matrix=shuffled_matrix,
        title=(
            "Shuffled-Prompt Control\n"
            "Incorrect Prompt Correspondence"
        ),
        colorbar_label="Shuffled cosine similarity",
        vmin=-shuffled_limit,
        vmax=shuffled_limit,
    )

    add_heatmap(
        axis=axes[2],
        matrix=difference_matrix,
        title=(
            "Real Minus Shuffled\n"
            "Prompt-Specific Alignment"
        ),
        colorbar_label="Similarity advantage",
        vmin=difference_vmin,
        vmax=difference_vmax,
    )

    figure.suptitle(
        "Llama-3.2-1B vs Llama-3.2-3B "
        "Held-out Procrustes Analysis",
        fontsize=16,
    )

    plt.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    plt.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    plt.close()

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

    print(f"Matrix shape: {real_matrix.shape}")
    print(
        f"Real plotting range: "
        f"{real_vmin:.6f} to {real_vmax:.6f}"
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

    print(f"\nSaved PNG to: {OUTPUT_PNG}")
    print(f"Saved PDF to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()