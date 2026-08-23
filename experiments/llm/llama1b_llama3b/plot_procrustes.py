from pathlib import Path

import numpy as np

from alignment.visualization import (
    plot_similarity_heatmap,
)


MATRIX_PATH = Path(
    "results/llama1b_llama3b/procrustes_matrix.npy"
)

OUTPUT_PNG = Path(
    "figures/llama1b_llama3b/procrustes_heatmap.png"
)

OUTPUT_PDF = Path(
    "figures/llama1b_llama3b/procrustes_heatmap.pdf"
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
            "Layer-wise Procrustes Similarity\n"
            "Llama-3.2-1B vs Llama-3.2-3B"
        ),
        x_label="Llama-3.2-3B layer",
        y_label="Llama-3.2-1B layer",
        colorbar_label="Procrustes similarity",
        vmin=0,
        vmax=1,
        mark_maximum=True,
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
