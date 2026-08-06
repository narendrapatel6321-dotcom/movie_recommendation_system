"""Central configuration for the neural recommender pipeline."""

from dataclasses import dataclass, field
from typing import Tuple

import torch

def _resolve_device() -> str:
    """Pick the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class RecommenderConfig:
    """Immutable hyperparameter registry for data, model, and training layers.

    All downstream components receive a single instance of this class so that
    no magic numbers are scattered across modules.

    Attributes:
        data_dir: Relative path to the directory containing MovieLens .dat files.
        results_dir: Directory where model checkpoints and plots are written.
        embed_dim: Embedding dimension for the MF model.
        mf_dim: GMF-path embedding dimension for NeuMF.
        mlp_embed: MLP-path embedding dimension for MLP and NeuMF.
        mlp_layers: Hidden-layer widths for MLP and the MLP branch of NeuMF.
        dropout: Dropout probability applied after each hidden layer.
        epochs: Maximum training epochs before forced termination.
        lr: Adam learning rate.
        batch_size: Mini-batch size for training DataLoader.
        num_negatives: Negative samples drawn per positive interaction during training.
        eval_negatives: Negative samples drawn per user during evaluation (standard = 99).
        eval_k: Cutoff rank K for HR@K and NDCG@K.
        patience: Early-stopping patience measured in validation epochs.
        seed: Global random seed for reproducibility.
        device: Compute device string ("cuda", "mps", or "cpu"), auto-resolved
            at construction time unless explicitly overridden.
    """

    data_dir: str = "data/ml-1m"
    results_dir: str = "results"

    embed_dim: int = 64
    mf_dim: int = 32
    mlp_embed: int = 32
    mlp_layers: Tuple[int, ...] = field(default_factory=lambda: (64, 32, 16))
    dropout: float = 0.2

    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 1024
    num_negatives: int = 4
    eval_negatives: int = 99
    eval_k: int = 10
    patience: int = 5

    seed: int = 42
    device: str = field(default_factory=_resolve_device)
