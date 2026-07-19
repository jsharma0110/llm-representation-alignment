from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LLAMA_1B_DIR = PROJECT_ROOT / "results" / "llama1b"
LLAMA_3B_DIR = PROJECT_ROOT / "results" / "llama3b"

OUTPUT_NPY = (
    PROJECT_ROOT
    / "results"
    / "procrustes_heldout_llama1b_llama3b.npy"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "procrustes_heldout_llama1b_llama3b.csv"
)

SHUFFLED_OUTPUT_NPY = (
    PROJECT_ROOT
    / "results"
    / "procrustes_heldout_shuffled_llama1b_llama3b.npy"
)

PCA_DIM = 64
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42


def layer_number(path: Path) -> int:
    """
    Extract the numerical layer index from a filename
    such as layer_12.pt.
    """
    match = re.fullmatch(r"layer_(\d+)\.pt", path.name)

    if match is None:
        raise ValueError(
            f"Unexpected layer filename: {path.name}"
        )

    return int(match.group(1))


def get_layer_files(directory: Path) -> list[Path]:
    """
    Return all layer files in numerical layer order.
    """
    files = sorted(
        directory.glob("layer_*.pt"),
        key=layer_number,
    )

    if not files:
        raise FileNotFoundError(
            f"No layer files found in {directory}"
        )

    return files


def load_tensor(path: Path) -> torch.Tensor:
    """
    Load one saved layer representation tensor.
    Expected shape: [number_of_samples, hidden_dimension].
    """
    tensor = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    ).float()

    if tensor.ndim != 2:
        raise ValueError(
            f"{path} should contain a 2D tensor, "
            f"but found shape {tuple(tensor.shape)}"
        )

    return tensor


def fit_pca(
    train: torch.Tensor,
    test: torch.Tensor,
    dimension: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Fit PCA using only the training samples.

    Returns:
        reduced_train
        reduced_test
        training_mean
        PCA directions
    """
    train_mean = train.mean(dim=0, keepdim=True)

    centered_train = train - train_mean
    centered_test = test - train_mean

    actual_dimension = min(
        dimension,
        centered_train.shape[0] - 1,
        centered_train.shape[1],
    )

    _, _, directions = torch.pca_lowrank(
        centered_train,
        q=actual_dimension,
        center=False,
    )

    reduced_train = centered_train @ directions
    reduced_test = centered_test @ directions

    return (
        reduced_train,
        reduced_test,
        train_mean,
        directions,
    )


def fit_orthogonal_map(
    source_train: torch.Tensor,
    target_train: torch.Tensor,
) -> torch.Tensor:
    """
    Fit the orthogonal transformation that best maps the
    source training representations to the target training
    representations.
    """
    if source_train.shape != target_train.shape:
        raise ValueError(
            "Training representations must have matching shapes. "
            f"Received {tuple(source_train.shape)} and "
            f"{tuple(target_train.shape)}."
        )

    cross_covariance = source_train.T @ target_train

    u, _, vh = torch.linalg.svd(
        cross_covariance,
        full_matrices=False,
    )

    transformation = u @ vh

    return transformation


def heldout_similarity(
    source_train: torch.Tensor,
    source_test: torch.Tensor,
    target_train: torch.Tensor,
    target_test: torch.Tensor,
) -> float:
    """
    Fit the Procrustes transformation on the training samples
    and evaluate it on held-out test samples.

    Similarity is measured as cosine similarity between the
    flattened aligned source test matrix and target test matrix.
    """
    transformation = fit_orthogonal_map(
        source_train,
        target_train,
    )

    aligned_source_test = source_test @ transformation

    source_vector = aligned_source_test.flatten()
    target_vector = target_test.flatten()

    similarity = F.cosine_similarity(
        source_vector,
        target_vector,
        dim=0,
    )

    return float(similarity)


def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    llama1b_files = get_layer_files(LLAMA_1B_DIR)
    llama3b_files = get_layer_files(LLAMA_3B_DIR)

    first_llama1b_tensor = load_tensor(
        llama1b_files[0]
    )
    first_llama3b_tensor = load_tensor(
        llama3b_files[0]
    )

    number_of_samples = first_llama1b_tensor.shape[0]

    if first_llama3b_tensor.shape[0] != number_of_samples:
        raise ValueError(
            "The two model result folders contain different "
            "numbers of samples."
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

    print(f"Samples: {number_of_samples}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print(f"PCA dimension: {PCA_DIM}")
    print(f"Llama-1B layers: {len(llama1b_files)}")
    print(f"Llama-3B layers: {len(llama3b_files)}")

    llama1b_reduced = []
    llama3b_reduced = []

    print("\nPreparing Llama-1B layers...")

    for path in llama1b_files:
        representations = load_tensor(path)

        if representations.shape[0] != number_of_samples:
            raise ValueError(
                f"{path} has an unexpected number of samples."
            )

        train, test, _, _ = fit_pca(
            representations[train_indices],
            representations[test_indices],
            PCA_DIM,
        )

        llama1b_reduced.append(
            (train, test)
        )

        print(
            f"  Prepared layer {layer_number(path)}"
        )

    print("\nPreparing Llama-3B layers...")

    for path in llama3b_files:
        representations = load_tensor(path)

        if representations.shape[0] != number_of_samples:
            raise ValueError(
                f"{path} has an unexpected number of samples."
            )

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

        print(
            f"  Prepared layer {layer_number(path)}"
        )

    matrix = np.zeros(
        (
            len(llama1b_files),
            len(llama3b_files),
        ),
        dtype=np.float32,
    )

    shuffled_matrix = np.zeros_like(matrix)

    print("\nComputing held-out similarities...")

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

            shuffled_matrix[i, j] = heldout_similarity(
                source_train,
                source_test,
                target_train,
                shuffled_target_test,
            )

        print(
            f"  Finished Llama-1B layer "
            f"{layer_number(llama1b_files[i])}"
        )

    OUTPUT_NPY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_NPY,
        matrix,
    )

    np.save(
        SHUFFLED_OUTPUT_NPY,
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

    dataframe = pd.DataFrame(
        matrix,
        index=row_names,
        columns=column_names,
    )

    dataframe.to_csv(
        OUTPUT_CSV
    )

    best_position = np.unravel_index(
        np.argmax(matrix),
        matrix.shape,
    )

    best_llama1b_layer = layer_number(
        llama1b_files[best_position[0]]
    )

    best_llama3b_layer = layer_number(
        llama3b_files[best_position[1]]
    )

    print("\nDone.")
    print(f"Matrix shape: {matrix.shape}")
    print(
        f"Minimum real similarity: "
        f"{matrix.min():.6f}"
    )
    print(
        f"Maximum real similarity: "
        f"{matrix.max():.6f}"
    )
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
    print(
        "Best layer pair: "
        f"Llama-1B layer {best_llama1b_layer} and "
        f"Llama-3B layer {best_llama3b_layer}"
    )
    print(
        f"Saved real matrix to: "
        f"{OUTPUT_NPY}"
    )
    print(
        f"Saved CSV matrix to: "
        f"{OUTPUT_CSV}"
    )
    print(
        f"Saved shuffled control to: "
        f"{SHUFFLED_OUTPUT_NPY}"
    )


if __name__ == "__main__":
    main()