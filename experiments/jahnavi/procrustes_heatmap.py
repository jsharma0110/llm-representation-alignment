from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATRIX_PATH = (
    PROJECT_ROOT
    / "results"
    / "procrustes_matrix_llama1b_llama3b.npy"
)

OUTPUT_PNG = (
    PROJECT_ROOT
    / "figures"
    / "procrustes_heatmap_llama1b_llama3b.png"
)

OUTPUT_PDF = (
    PROJECT_ROOT
    / "figures"
    / "procrustes_heatmap_llama1b_llama3b.pdf"
)


def main() -> None:
    matrix = np.load(MATRIX_PATH)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 7))

    image = plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Procrustes similarity",
    )

    plt.xlabel("Llama-3.2-3B layer")
    plt.ylabel("Llama-3.2-1B layer")
    plt.title(
        "Layer-wise Procrustes Similarity\n"
        "Llama-3.2-1B vs Llama-3.2-3B"
    )

    plt.xticks(
        ticks=np.arange(matrix.shape[1]),
        labels=np.arange(matrix.shape[1]),
        rotation=90,
    )

    plt.yticks(
        ticks=np.arange(matrix.shape[0]),
        labels=np.arange(matrix.shape[0]),
    )

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
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.savefig(OUTPUT_PDF)
    plt.close()

    print(f"Loaded matrix with shape: {matrix.shape}")
    print(f"Saved PNG to: {OUTPUT_PNG}")
    print(f"Saved PDF to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()