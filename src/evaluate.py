"""Ranking evaluation: Hit Rate@K and NDCG@K under the leave-one-out protocol."""

import logging
from typing import Dict, List

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class RankingEvaluator:
    """Computes Hit Rate@K and NDCG@K under the standard NCF evaluation protocol.

    Each evaluation sample contains one positive item at index 0 followed by
    ``num_negatives`` unobserved items.  The model scores all candidates; the
    positive is considered "hit" if it appears within the top-K ranked positions.

    Args:
        k: Cutoff rank for both HR@K and NDCG@K (default 10).
        eval_batch_size: Number of users scored per forward pass.
    """

    def __init__(self, k: int = 10, eval_batch_size: int = 256) -> None:
        self.k: int = k
        self.eval_batch_size: int = eval_batch_size

    def evaluate(self, model: tf.keras.Model, eval_dataset) -> Dict[str, float]:
        """Score every user in ``eval_dataset`` and aggregate HR@K and NDCG@K.

        Args:
            model: A trained recommendation model with a
                ``call((user, item), training=False)`` signature returning
                interaction probability tensors.
            eval_dataset: An ``EvalDataset`` (TF version, from ``data.py``)
                where the positive item occupies column 0 of ``candidates``.

        Returns:
            Dict with keys ``"hr@{k}"`` and ``"ndcg@{k}"`` mapping to floats.
        """
        hrs: List[float] = []
        ndcgs: List[float] = []

        ds = eval_dataset.as_tf_dataset(batch_size=self.eval_batch_size)

        try:
            for user, items in ds:
                # user: (B,)   items: (B, C) where C = 1 + num_negatives,
                # positive item is always at column 0.
                batch_size = tf.shape(items)[0]
                num_candidates = tf.shape(items)[1]

                user_flat = tf.repeat(user, num_candidates)
                items_flat = tf.reshape(items, [-1])

                scores_flat = model((user_flat, items_flat), training=False)
                scores: np.ndarray = tf.reshape(
                    scores_flat, [batch_size, num_candidates]
                ).numpy()

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
