"""Ranking evaluation: Hit Rate@K and NDCG@K under the leave-one-out protocol."""

import logging
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class RankingEvaluator:
    """Computes Hit Rate@K and NDCG@K under the standard NCF evaluation protocol.

    Each evaluation sample contains one positive item at index 0 followed by
    ``num_negatives`` unobserved items.  The model scores all candidates; the
    positive is considered "hit" if it appears within the top-K ranked positions.

    Args:
        k: Cutoff rank for both HR@K and NDCG@K (default 10).
    """

    def __init__(self, k: int = 10, eval_batch_size: int = 256) -> None:
        self.k: int = k
        self.eval_batch_size: int = eval_batch_size

    def evaluate(
        self,
        model: torch.nn.Module,
        eval_dataset: Dataset,
        device: str,
    ) -> Dict[str, float]:
        """Score every user in ``eval_dataset`` and aggregate HR@K and NDCG@K.

        Args:
            model: A trained recommendation model with a ``forward(user, item)``
                signature returning interaction probability tensors.
            eval_dataset: An ``EvalDataset`` instance where the positive item
                occupies index 0 among the candidate list.
            device: Torch device string (``"cpu"``, ``"cuda"``, or ``"mps"``).

        Returns:
            Dict with keys ``"hr@{k}"`` and ``"ndcg@{k}"`` mapping to floats.
        """
        model.eval()
        hrs: List[float] = []
        ndcgs: List[float] = []

        loader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=False)

        try:
            with torch.no_grad():
                for user, items in loader:
                    # user: (B,)   items: (B, C) where C = 1 + num_negatives,
                    # positive item is always at column 0.
                    batch_size, num_candidates = items.shape

                    user_flat = user.to(device).repeat_interleave(num_candidates)
                    items_flat = items.reshape(-1).to(device)

                    scores_flat = model(user_flat, items_flat)
                    scores: np.ndarray = scores_flat.view(batch_size, num_candidates).cpu().numpy()

                    # 0-indexed rank of the positive = count of candidates scoring higher.
                    pos_scores = scores[:, 0:1]
                    ranks: np.ndarray = (scores > pos_scores).sum(axis=1)

                    hits = (ranks < self.k).astype(np.float64)
                    ndcg_vals = np.where(ranks < self.k, 1.0 / np.log2(ranks + 2), 0.0)

                    hrs.extend(hits.tolist())
                    ndcgs.extend(ndcg_vals.tolist())
        except Exception as exc:
            logger.error("Evaluation loop failed: %s", exc)
            raise

        metrics = {
            f"hr@{self.k}": float(np.mean(hrs)),
            f"ndcg@{self.k}": float(np.mean(ndcgs)),
        }
        logger.info("Evaluation complete — %s", metrics)
        return metrics


    @staticmethod
    def log_report(model_name: str, metrics: Dict[str, float]) -> None:
        """Emit a formatted metric summary through the logging system.

        Args:
            model_name: Human-readable model identifier for the report header.
            metrics: Dict of metric name → value as returned by ``evaluate()``.
        """
        separator = "=" * 40
        logger.info(separator)
        logger.info("  %s", model_name)
        logger.info(separator)
        for key, val in metrics.items():
            logger.info("  %s : %.4f", key.rjust(10), val)
        logger.info(separator)
