from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_similarity_heatmap(
    matrix: np.ndarray,
    output_png: str | Path,
    output_pdf: str | Path,
    title: str,
    x_label: str,
    y_label: str,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    mark_maximum: bool = True,
) -> None:
    """
    Plot and save a layer-wise similarity heatmap.
    """
    output_png = Path(output_png)
    output_pdf = Path(output_pdf)

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(13, 7))

    image = plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(
        image,
        label=colorbar_label,
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.xticks(
        ticks=np.arange(matrix.shape[1]),
        labels=np.arange(matrix.shape[1]),
        rotation=90,
    )

    plt.yticks(
        ticks=np.arange(matrix.shape[0]),
        labels=np.arange(matrix.shape[0]),
    )

    if mark_maximum:
        maximum_position = np.unravel_index(
            np.argmax(matrix),
            matrix.shape,
        )

        plt.scatter(
            maximum_position[1],
            maximum_position[0],
            marker="x",
            s=100,
            linewidths=2,
        )

    plt.tight_layout()

    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    plt.close()


def normalized_diagonal_score(
    matrix: np.ndarray,
    width: float = 0.1,
) -> tuple[float, float, float]:
    """
    Compare values near the normalized-depth diagonal
    against values away from it.
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
        row_depths[:, None]
        - column_depths[None, :]
    )

    diagonal_mask = (
        distance_from_diagonal <= width
    )

    off_diagonal_mask = (
        distance_from_diagonal > width
    )

    diagonal_mean = float(
        matrix[diagonal_mask].mean()
    )

    off_diagonal_mean = float(
        matrix[off_diagonal_mask].mean()
    )

    diagonal_advantage = (
        diagonal_mean
        - off_diagonal_mean
    )

    return (
        diagonal_mean,
        off_diagonal_mean,
        diagonal_advantage,
    )


def plot_heldout_comparison(
    real_matrix: np.ndarray,
    shuffled_matrix: np.ndarray,
    output_png: str | Path,
    output_pdf: str | Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """
    Plot real, shuffled, and real-minus-shuffled
    held-out similarity matrices.
    """
    if real_matrix.shape != shuffled_matrix.shape:
        raise ValueError(
            "The real and shuffled matrices "
            "must have matching shapes."
        )

    output_png = Path(output_png)
    output_pdf = Path(output_pdf)

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    difference_matrix = (
        real_matrix - shuffled_matrix
    )

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

    def add_heatmap(
        axis,
        matrix,
        subplot_title,
        colorbar_label,
        vmin,
        vmax,
    ):
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        axis.set_title(subplot_title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)

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

        colorbar.set_label(
            colorbar_label
        )

    add_heatmap(
        axes[0],
        real_matrix,
        "Held-out Procrustes Similarity\nReal Prompt Correspondence",
        "Held-out cosine similarity",
        real_vmin,
        real_vmax,
    )

    add_heatmap(
        axes[1],
        shuffled_matrix,
        "Shuffled-Prompt Control\nIncorrect Prompt Correspondence",
        "Shuffled cosine similarity",
        -shuffled_limit,
        shuffled_limit,
    )

    add_heatmap(
        axes[2],
        difference_matrix,
        "Real Minus Shuffled\nPrompt-Specific Alignment",
        "Similarity advantage",
        difference_vmin,
        difference_vmax,
    )

    figure.suptitle(
        title,
        fontsize=16,
    )

    plt.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    plt.close()
