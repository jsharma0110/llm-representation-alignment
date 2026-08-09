from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F


def layer_number(path: Path) -> int:
    """Extract numerical layer index from filenames like layer_12.pt."""
    match = re.fullmatch(r"layer_(\d+)\.pt", path.name)

    if match is None:
        raise ValueError(f"Unexpected layer filename: {path.name}")

    return int(match.group(1))


def get_layer_files(directory: str | Path) -> list[Path]:
    """Return saved layer tensors in numerical order."""
    directory = Path(directory)

    files = sorted(
        directory.glob("layer_*.pt"),
        key=layer_number,
    )

    if not files:
        raise FileNotFoundError(
            f"No layer files found in {directory}"
        )

    return files


def load_tensor(path: str | Path) -> torch.Tensor:
    """Load one 2D saved representation tensor."""
    path = Path(path)

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


def reduce_with_pca(
    representations: torch.Tensor,
    target_dim: int,
) -> torch.Tensor:
    """
    Center representations and reduce them using PCA.
    """
    representations = (
        representations
        - representations.mean(dim=0, keepdim=True)
    )

    maximum_dimension = min(
        representations.shape[0] - 1,
        representations.shape[1],
        target_dim,
    )

    _, _, directions = torch.pca_lowrank(
        representations,
        q=maximum_dimension,
        center=False,
    )

    return representations @ directions


def procrustes_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    """
    Compute normalized orthogonal Procrustes similarity.
    """
    if first.shape != second.shape:
        raise ValueError(
            "Reduced representations must have matching shapes. "
            f"Received {tuple(first.shape)} and "
            f"{tuple(second.shape)}."
        )

    first = (
        first
        / torch.linalg.norm(
            first,
            ord="fro",
        ).clamp_min(1e-12)
    )

    second = (
        second
        / torch.linalg.norm(
            second,
            ord="fro",
        ).clamp_min(1e-12)
    )

    cross_covariance = first.T @ second
    singular_values = torch.linalg.svdvals(
        cross_covariance
    )

    similarity = singular_values.sum()

    return float(
        similarity.clamp(min=0.0, max=1.0)
    )


def compute_procrustes_matrix(
    model_a_dir: str | Path,
    model_b_dir: str | Path,
    pca_dim: int = 128,
) -> tuple[np.ndarray, list[Path], list[Path]]:
    """
    Compute pairwise Procrustes similarity across model layers.
    """
    model_a_files = get_layer_files(model_a_dir)
    model_b_files = get_layer_files(model_b_dir)

    model_a_reduced = [
        reduce_with_pca(
            load_tensor(path),
            pca_dim,
        )
        for path in model_a_files
    ]

    model_b_reduced = [
        reduce_with_pca(
            load_tensor(path),
            pca_dim,
        )
        for path in model_b_files
    ]

    matrix = np.zeros(
        (
            len(model_a_reduced),
            len(model_b_reduced),
        ),
        dtype=np.float32,
    )

    for i, first in enumerate(model_a_reduced):
        for j, second in enumerate(model_b_reduced):
            matrix[i, j] = procrustes_similarity(
                first,
                second,
            )

    return matrix, model_a_files, model_b_files


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
    Fit PCA using training samples only.
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
    """Fit an orthogonal transformation from source to target."""
    if source_train.shape != target_train.shape:
        raise ValueError(
            "Training representations must have matching shapes. "
            f"Received {tuple(source_train.shape)} and "
            f"{tuple(target_train.shape)}."
        )

    cross_covariance = (
        source_train.T @ target_train
    )

    u, _, vh = torch.linalg.svd(
        cross_covariance,
        full_matrices=False,
    )

    return u @ vh


def heldout_similarity(
    source_train: torch.Tensor,
    source_test: torch.Tensor,
    target_train: torch.Tensor,
    target_test: torch.Tensor,
) -> float:
    """
    Fit alignment on train samples and evaluate on held-out samples.
    """
    transformation = fit_orthogonal_map(
        source_train,
        target_train,
    )

    aligned_source_test = (
        source_test @ transformation
    )

    similarity = F.cosine_similarity(
        aligned_source_test.flatten(),
        target_test.flatten(),
        dim=0,
    )

    return float(similarity)
