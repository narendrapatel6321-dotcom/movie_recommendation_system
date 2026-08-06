"""Pure MLP-based neural collaborative filtering model."""

from typing import Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    """NCF-MLP: concatenated user/item embeddings fed through a fully-connected tower.

    The user embedding p_u and item embedding q_i are concatenated into a single
    vector [p_u || q_i], which is then passed through a stack of linear→ReLU→dropout
    layers.  A final linear projection maps to a scalar interaction score normalised
    by sigmoid.

    Args:
        n_users: Total number of users.
        n_items: Total number of items.
        embed_dim: Dimensionality of each user/item embedding (concatenated input = 2×embed_dim).
        layers: Sequence of hidden-layer widths for the MLP tower.
        dropout: Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 32,
        layers: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.user_emb: nn.Embedding = nn.Embedding(n_users, embed_dim)
        self.item_emb: nn.Embedding = nn.Embedding(n_items, embed_dim)

        in_dim: int = embed_dim * 2
        mlp_layers = []
        for out_dim in layers:
            mlp_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim

        self.mlp: nn.Sequential = nn.Sequential(*mlp_layers)
        self.output: nn.Linear = nn.Linear(in_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise embeddings with small Gaussian noise; all Linear layers with Xavier uniform."""
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.output.weight)


    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid-normalised interaction scores for a batch.

        Args:
            user: Long tensor of user indices, shape (B,).
            item: Long tensor of item indices, shape (B,).

        Returns:
            Float tensor of predicted interaction probabilities, shape (B,).
        """
        x: torch.Tensor = torch.cat([self.user_emb(user), self.item_emb(item)], dim=-1)
        return torch.sigmoid(self.output(self.mlp(x)).squeeze(-1))
