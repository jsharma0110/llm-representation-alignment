from pathlib import Path

import numpy as np

from alignment.visualization import (
    plot_similarity_heatmap,
)


MATRIX_PATH = Path(
    "results/tinyllama_qwen/cka_matrix.npy"
)

OUTPUT_PNG = Path(
    "figures/tinyllama_qwen/cka_heatmap.png"
)

OUTPUT_PDF = Path(
    "figures/tinyllama_qwen/cka_heatmap.pdf"
)


def main():
    matrix = np.load(
        MATRIX_PATH
    )

    plot_similarity_heatmap(
        matrix=matrix,
        output_png=OUTPUT_PNG,
        output_pdf=OUTPUT_PDF,
        title="Layer-wise CKA Similarity",
        x_label="Qwen Layer",
        y_label="TinyLlama Layer",
        colorbar_label="CKA Similarity",
        layer_prefix="L",
        figsize=(10, 8),
        dpi=600,
    )

    print(
        f"Loaded matrix with shape: "
        f"{matrix.shape}"
    )

    print(
        f"Saved PNG to: {OUTPUT_PNG}"
    )

    print(
        f"Saved PDF to: {OUTPUT_PDF}"
    )


if __name__ == "__main__":
    main()
