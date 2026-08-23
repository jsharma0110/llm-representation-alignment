import torch


def mean_pool(hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Mean pool across the sequence dimension.

    Args:
        hidden_state:
            Tensor of shape (batch_size, sequence_length, hidden_dim).

    Returns:
        Tensor of shape (batch_size, hidden_dim).
    """
    return hidden_state.mean(dim=1)


def pool_hidden_state(
    hidden_state: torch.Tensor,
    strategy: str = "mean",
) -> torch.Tensor:
    """
    Pool token-level hidden states into one representation.

    Args:
        hidden_state:
            Tensor of shape (batch_size, sequence_length, hidden_dim).
        strategy:
            Pooling strategy. Currently supports "mean".

    Returns:
        Pooled tensor of shape (batch_size, hidden_dim).
    """
    if strategy == "mean":
        return mean_pool(hidden_state)

    raise ValueError(
        f"Unsupported pooling strategy: {strategy}"
    )
