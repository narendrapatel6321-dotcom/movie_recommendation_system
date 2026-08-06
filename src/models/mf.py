"""Matrix Factorization model for implicit feedback recommendation."""

import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """Classical dot-product Matrix Factorization with per-user and per-item biases.

    Scores a (user, item) pair as:
        sigmoid( <p_u, q_i> + b_u + b_i )

    where p_u and q_i are learned embedding vectors and b_u, b_i are scalar
    bias terms.  Output is passed through sigmoid so BCE loss can be applied
    directly for implicit-feedback training.

    Args:
        n_users: Total number of users in the dataset.
        n_items: Total number of items in the dataset.
        embed_dim: Dimensionality of user and item embedding vectors.
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.user_emb: nn.Embedding = nn.Embedding(n_users, embed_dim)
        self.item_emb: nn.Embedding = nn.Embedding(n_items, embed_dim)
        self.user_bias: nn.Embedding = nn.Embedding(n_users, 1)
        self.item_bias: nn.Embedding = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid-normalised interaction scores for a batch.

        Args:
            user: Long tensor of user indices, shape (B,).
            item: Long tensor of item indices, shape (B,).

        Returns:
            Float tensor of predicted interaction probabilities, shape (B,).
        """
        pu: torch.Tensor = self.user_emb(user)
        qi: torch.Tensor = self.item_emb(item)
        bu: torch.Tensor = self.user_bias(user).squeeze(-1)
        bi: torch.Tensor = self.item_bias(item).squeeze(-1)
        return torch.sigmoid((pu * qi).sum(dim=-1) + bu + bi)
