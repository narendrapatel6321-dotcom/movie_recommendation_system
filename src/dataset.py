"""PyTorch Dataset implementations for training and evaluation."""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    """Implicit-feedback training set with inline negative sampling.

    For every observed (user, item) interaction one positive sample is
    emitted alongside ``num_negatives`` uniformly drawn unobserved items.
    Negatives are sampled once at construction time (fixed across epochs).

    Args:
        interactions: DataFrame with columns ``user`` and ``item``.
        n_items: Total number of distinct items in the catalogue.
        user_pos_items: Mapping from user index to the full set of items the
            user has interacted with (used to exclude known positives from
            negative sampling).
        num_negatives: Number of negative items to pair with each positive.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        n_items: int,
        user_pos_items: Dict[int, Set[int]],
        num_negatives: int = 4,
        seed: int = 42,
    ) -> None:
        self.n_items: int = n_items
        self.user_pos_items: Dict[int, Set[int]] = user_pos_items
        self.num_negatives: int = num_negatives
	self.seed: int = seed
        self._data: List[Tuple[int, int, float]] = self._generate(interactions)
        logger.info(
            "TrainDataset: %d samples (%d positives × %d negatives each)",
            len(self._data),
            len(interactions),
            num_negatives,
        )

    def _generate(self, interactions: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """Pre-generate the full list of (user, item, label) triples.

        Args:
            interactions: Positive interaction DataFrame.

        Returns:

            List of (user_idx, item_idx, label) tuples where label ∈ {0.0, 1.0}.
        """

        data: List[Tuple[int, int, float]] = []
        rng = np.random.default_rng(self.seed)
        max_attempts: int = self.n_items * 10

        for row in interactions.itertuples(index=False):
            u, i = int(row.user), int(row.item)
            data.append((u, i, 1.0))

            pos_set: Set[int] = self.user_pos_items[u]
            sampled: int = 0
            attempts: int = 0
            while sampled < self.num_negatives:
                attempts += 1
                if attempts > max_attempts:
                    logger.warning(
                        "User %d: exhausted %d attempts sampling negatives "
                        "(%d/%d found). Item catalogue may be too small/dense "
                        "relative to num_negatives.",
                        u, max_attempts, sampled, self.num_negatives,
                    )
                    break
                j = int(rng.integers(self.n_items))
                if j not in pos_set:
                    data.append((u, j, 0.0))
                    sampled += 1

        return data

    def __len__(self) -> int:
        """Return the total number of (user, item, label) triples."""
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a single (user, item, label) triple as tensors.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (user_tensor, item_tensor, label_tensor).
        """
        u, i, label = self._data[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )


class EvalDataset(Dataset):
    """Per-user evaluation set following the leave-one-out ranking protocol.

    Each sample contains one positive item (at index 0) followed by
    ``num_negatives`` randomly drawn unobserved items.  Callers score all
    candidates and report HR@K / NDCG@K.

    Args:
        pos_interactions: DataFrame with the held-out positive (user, item) pairs.
        user_pos_items: Full set of each user's observed items (for exclusion).
        n_items: Total catalogue size.
        num_negatives: Number of negative candidates per evaluation sample (default 99).
    """

    def __init__(
        self,
        pos_interactions: pd.DataFrame,
        user_pos_items: Dict[int, Set[int]],
        n_items: int,
        num_negatives: int = 99,
        seed: int = 0,
    ) -> None:
        self._samples: List[Tuple[int, List[int]]] = []
        rng = np.random.default_rng(seed)
        max_attempts: int = n_items * 10

        for row in pos_interactions.itertuples(index=False):
            u, pos_i = int(row.user), int(row.item)

            pos_set: Set[int] = user_pos_items[u]
            negs: List[int] = []
            attempts: int = 0

            while len(negs) < num_negatives:
                attempts += 1
                if attempts > max_attempts:
                    logger.warning(
                        "User %d: exhausted %d attempts sampling eval negatives "
                        "(%d/%d found).",
                        u, max_attempts, len(negs), num_negatives,
                    )
                    break
                j = int(rng.integers(n_items))
                if j not in pos_set:
                    negs.append(j)

            self._samples.append((u, [pos_i] + negs))

        logger.info(
            "EvalDataset: %d users × %d candidates each",
            len(self._samples),
            1 + num_negatives,
        )

    def __len__(self) -> int:
        """Return the number of evaluation users."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (user_tensor, candidates_tensor) for one evaluation user.

        The positive item is always at position 0 in ``candidates_tensor``.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (user_tensor shape (), candidates_tensor shape (1 + num_negatives,)).
        """
        u, items = self._samples[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
        )
