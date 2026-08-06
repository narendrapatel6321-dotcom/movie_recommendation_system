"""Neural Matrix Factorization combining GMF and MLP paths (He et al. 2017)."""

from typing import Tuple

import torch
import torch.nn as nn


class NeuMF(nn.Module):
    """Neural Matrix Factorization: fused Generalised MF and MLP branches.

    Architecture (He et al., 2017 — "Neural Collaborative Filtering"):

    GMF branch:
        p_u_mf  ⊙  q_i_mf   →  element-wise product vector of shape (mf_dim,)

    MLP branch:
        [p_u_mlp || q_i_mlp]  →  FC(64)→ReLU→FC(32)→ReLU→FC(16)→ReLU

    Fusion:
        [gmf_out || mlp_out]  →  Linear(mf_dim + last_mlp_dim, 1)  →  sigmoid

    Separate embedding matrices per branch allow each path to learn a
    complementary representation: GMF captures bilinear interactions while
    MLP captures non-linear patterns.

    Args:
        n_users: Total number of users.
        n_items: Total number of items.
        mf_dim: Embedding dimension for the GMF branch.
        mlp_embed: Embedding dimension for the MLP branch (concatenated = 2×mlp_embed).
        layers: Hidden-layer widths for the MLP tower.
        dropout: Dropout probability after each hidden activation.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        mf_dim: int = 32,
        mlp_embed: int = 32,
        layers: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.mf_user: nn.Embedding = nn.Embedding(n_users, mf_dim)
        self.mf_item: nn.Embedding = nn.Embedding(n_items, mf_dim)
        self.mlp_user: nn.Embedding = nn.Embedding(n_users, mlp_embed)
        self.mlp_item: nn.Embedding = nn.Embedding(n_items, mlp_embed)

        in_dim: int = mlp_embed * 2
        mlp_layers = []
        for out_dim in layers:
            mlp_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim

        self.mlp: nn.Sequential = nn.Sequential(*mlp_layers)
        self.output: nn.Linear = nn.Linear(mf_dim + in_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise all embeddings with N(0, 0.01); all Linear layers with Xavier uniform."""
        for emb in (self.mf_user, self.mf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Compute fused GMF+MLP interaction scores for a batch.

        Args:
            user: Long tensor of user indices, shape (B,).
            item: Long tensor of item indices, shape (B,).

        Returns:
            Float tensor of predicted interaction probabilities, shape (B,).
        """
        gmf_out: torch.Tensor = self.mf_user(user) * self.mf_item(item)

        mlp_in: torch.Tensor = torch.cat(
            [self.mlp_user(user), self.mlp_item(item)], dim=-1
        )
        mlp_out: torch.Tensor = self.mlp(mlp_in)

        fused: torch.Tensor = torch.cat([gmf_out, mlp_out], dim=-1)
        return torch.sigmoid(self.output(fused).squeeze(-1))
