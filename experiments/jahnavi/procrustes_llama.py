from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LLAMA_1B_DIR = PROJECT_ROOT / "results" / "llama1b"
LLAMA_3B_DIR = PROJECT_ROOT / "results" / "llama3b"

OUTPUT_NPY = PROJECT_ROOT / "results" / "procrustes_matrix_llama1b_llama3b.npy"
OUTPUT_CSV = PROJECT_ROOT / "results" / "procrustes_matrix_llama1b_llama3b.csv"

# Both representation matrices will be reduced to this dimension.
# 128 is fast and safely below the 500-sample limit.
PCA_DIM = 128


def layer_number(path: Path) -> int:
    """Extract the numerical layer index from a filename such as layer_12.pt."""
    match = re.fullmatch(r"layer_(\d+)\.pt", path.name)

    if match is None:
        raise ValueError(f"Unexpected layer filename: {path.name}")

    return int(match.group(1))


def get_layer_files(directory: Path) -> list[Path]:
    """Return layer files in numerical order."""
    files = list(directory.glob("layer_*.pt"))

    if not files:
        raise FileNotFoundError(f"No layer files found in {directory}")

    return sorted(files, key=layer_number)


def load_and_reduce(path: Path, target_dim: int) -> torch.Tensor:
    """
    Load a [samples, hidden_dimension] tensor, center it,
    and reduce it using PCA.
    """
    representations = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    ).float()

    if representations.ndim != 2:
        raise ValueError(
            f"{path} should contain a 2D tensor, "
            f"but found shape {tuple(representations.shape)}"
        )

    # Center each hidden-state feature.
    representations = representations - representations.mean(
        dim=0,
        keepdim=True,
    )

    maximum_dimension = min(
        representations.shape[0] - 1,
        representations.shape[1],
        target_dim,
    )

    # Low-rank PCA. V contains the principal directions.
    _, _, principal_directions = torch.pca_lowrank(
        representations,
        q=maximum_dimension,
        center=False,
    )

    reduced = representations @ principal_directions
    return reduced


def procrustes_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    """
    Compute normalized orthogonal Procrustes similarity.

    A value close to 1 means that the two representations can be
    closely aligned using an orthogonal transformation.
    """
    if first.shape != second.shape:
        raise ValueError(
            "Reduced representations must have matching shapes. "
            f"Received {tuple(first.shape)} and {tuple(second.shape)}."
        )

    first = first / torch.linalg.norm(first, ord="fro").clamp_min(1e-12)
    second = second / torch.linalg.norm(second, ord="fro").clamp_min(1e-12)

    cross_covariance = first.T @ second
    singular_values = torch.linalg.svdvals(cross_covariance)

    similarity = singular_values.sum()
    return float(similarity.clamp(min=0.0, max=1.0))


def main() -> None:
    torch.manual_seed(42)

    llama1b_files = get_layer_files(LLAMA_1B_DIR)
    llama3b_files = get_layer_files(LLAMA_3B_DIR)

    print(f"Llama-1B layers: {len(llama1b_files)}")
    print(f"Llama-3B layers: {len(llama3b_files)}")
    print(f"PCA dimension: {PCA_DIM}")

    print("\nReducing Llama-1B representations...")
    llama1b_representations = []

    for path in llama1b_files:
        print(f"  Loading {path}")
        llama1b_representations.append(
            load_and_reduce(path, PCA_DIM)
        )

    print("\nReducing Llama-3B representations...")
    llama3b_representations = []

    for path in llama3b_files:
        print(f"  Loading {path}")
        llama3b_representations.append(
            load_and_reduce(path, PCA_DIM)
        )

    similarity_matrix = np.zeros(
        (
            len(llama1b_representations),
            len(llama3b_representations),
        ),
        dtype=np.float32,
    )

    print("\nComputing layer-wise Procrustes similarity...")

    for i, first in enumerate(llama1b_representations):
        for j, second in enumerate(llama3b_representations):
            similarity_matrix[i, j] = procrustes_similarity(
                first,
                second,
            )

        print(
            f"  Finished Llama-1B layer "
            f"{layer_number(llama1b_files[i])}"
        )

    OUTPUT_NPY.parent.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_NPY, similarity_matrix)

    row_labels = [
        f"llama1b_layer_{layer_number(path)}"
        for path in llama1b_files
    ]
    column_labels = [
        f"llama3b_layer_{layer_number(path)}"
        for path in llama3b_files
    ]

    dataframe = pd.DataFrame(
        similarity_matrix,
        index=row_labels,
        columns=column_labels,
    )
    dataframe.to_csv(OUTPUT_CSV)

    maximum_position = np.unravel_index(
        np.argmax(similarity_matrix),
        similarity_matrix.shape,
    )

    best_1b_layer = layer_number(
        llama1b_files[maximum_position[0]]
    )
    best_3b_layer = layer_number(
        llama3b_files[maximum_position[1]]
    )

    print("\nDone.")
    print(f"Matrix shape: {similarity_matrix.shape}")
    print(f"Minimum similarity: {similarity_matrix.min():.4f}")
    print(f"Maximum similarity: {similarity_matrix.max():.4f}")
    print(
        "Highest-similarity pair: "
        f"Llama-1B layer {best_1b_layer} and "
        f"Llama-3B layer {best_3b_layer}"
    )
    print(f"Saved NumPy matrix to: {OUTPUT_NPY}")
    print(f"Saved CSV matrix to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()