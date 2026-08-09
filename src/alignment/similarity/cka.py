from pathlib import Path

import numpy as np
import torch


def center_gram(K: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix."""
    n = K.shape[0]

    H = torch.eye(
        n,
        device=K.device,
        dtype=K.dtype,
    ) - torch.ones(
        (n, n),
        device=K.device,
        dtype=K.dtype,
    ) / n

    return H @ K @ H


def linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    standardize: bool = False,
) -> float:
    """
    Compute linear Centered Kernel Alignment (CKA).

    Args:
        X: Tensor of shape (num_examples, hidden_dim_x).
        Y: Tensor of shape (num_examples, hidden_dim_y).
        standardize: Whether to z-score features before computing CKA.

    Returns:
        Linear CKA similarity score.
    """
    X = X.float()
    Y = Y.float()

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must contain the same number of examples. "
            f"Got {X.shape[0]} and {Y.shape[0]}."
        )

    if standardize:
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        X = X / (X.std(dim=0, keepdim=True) + 1e-8)
        Y = Y / (Y.std(dim=0, keepdim=True) + 1e-8)

    K = center_gram(X @ X.T)
    L = center_gram(Y @ Y.T)

    hsic = (K * L).sum()

    norm_x = torch.sqrt((K * K).sum())
    norm_y = torch.sqrt((L * L).sum())

    denominator = norm_x * norm_y

    if denominator == 0:
        return 0.0

    return (hsic / denominator).item()


def get_layer_files(directory: str | Path) -> list[Path]:
    """Return saved layer tensors ordered by layer number."""
    directory = Path(directory)

    files = list(directory.glob("layer_*.pt"))

    return sorted(
        files,
        key=lambda path: int(path.stem.split("_")[-1]),
    )


def compute_cka_matrix(
    model_a_dir: str | Path,
    model_b_dir: str | Path,
    standardize: bool = False,
) -> np.ndarray:
    """
    Compute pairwise CKA between all saved layers of two models.
    """
    model_a_layers = get_layer_files(model_a_dir)
    model_b_layers = get_layer_files(model_b_dir)

    if not model_a_layers:
        raise FileNotFoundError(
            f"No layer_*.pt files found in {model_a_dir}"
        )

    if not model_b_layers:
        raise FileNotFoundError(
            f"No layer_*.pt files found in {model_b_dir}"
        )

    similarity = np.zeros(
        (len(model_a_layers), len(model_b_layers))
    )

    for i, model_a_file in enumerate(model_a_layers):
        X = torch.load(model_a_file, map_location="cpu")

        for j, model_b_file in enumerate(model_b_layers):
            Y = torch.load(model_b_file, map_location="cpu")

            score = linear_cka(
                X,
                Y,
                standardize=standardize,
            )

            similarity[i, j] = score

            print(
                f"{model_a_file.name} vs "
                f"{model_b_file.name}: {score:.4f}"
            )

    return similarity
