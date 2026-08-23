from pathlib import Path

import numpy as np

from alignment.visualization import (
    plot_similarity_heatmap,
)


MATRIX_PATH = Path(
    "results/llama1b_llama3b/cka_matrix.npy"
)

OUTPUT_PNG = Path(
    "figures/llama1b_llama3b/cka_heatmap.png"
)

OUTPUT_PDF = Path(
    "figures/llama1b_llama3b/cka_heatmap.pdf"
)


def main():
    matrix = np.load(
        MATRIX_PATH
    )

    plot_similarity_heatmap(
        matrix=matrix,
        output_png=OUTPUT_PNG,
        output_pdf=OUTPUT_PDF,
        title=(
            "Layer-wise Linear CKA Similarity\n"
            "Llama-3.2-1B vs Llama-3.2-3B"
        ),
        x_label="Llama-3.2-3B Layer",
        y_label="Llama-3.2-1B Layer",
        colorbar_label="Linear CKA Similarity",
        layer_prefix="L",
        figsize=(11, 9),
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
