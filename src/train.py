"""Training loop built on Keras's `model.fit()`, with ranking-based validation and early stopping."""

import logging
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from .evaluate import RankingEvaluator

logger = logging.getLogger(__name__)


class _RankingEvalCallback(tf.keras.callbacks.Callback):
    """Runs HR@K/NDCG@K validation at the end of every epoch and injects results into `logs`.

    Keras's built-in metrics only cover things computable from a per-batch
    loss/metric function (e.g. BCE), not a whole-dataset ranking pass over
    positive+negative candidates. This callback bridges that gap by running
    `RankingEvaluator` at epoch end and writing `val_hr`/`val_ndcg` into the
    shared `logs` dict, so downstream callbacks (`EarlyStopping`, `History`)
    can see and act on them as if they were native Keras metrics.

    Must be listed *before* `EarlyStopping` in `model.fit(callbacks=[...])`
    — Keras calls each callback's `on_epoch_end` in list order against the
    same `logs` object, so this callback's writes must happen first.

    Args:
        evaluator: A configured `RankingEvaluator`.
        val_dataset: `EvalDataset` (TF version, from `data.py`) for validation scoring.
    """

    def __init__(self, evaluator: RankingEvaluator, val_dataset) -> None:
        super().__init__()
        self._evaluator = evaluator
        self._val_dataset = val_dataset

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Compute validation ranking metrics and write them into `logs`.

        Args:
            epoch: Current epoch index (0-based, supplied by Keras).
            logs: Mutable dict of epoch metrics shared across callbacks;
                mutated in place so later callbacks see `val_hr`/`val_ndcg`.
        """
        logs = {} if logs is None else logs
        metrics = self._evaluator.evaluate(self.model, self._val_dataset)
        hr_key = f"hr@{self._evaluator.k}"
        ndcg_key = f"ndcg@{self._evaluator.k}"
        logs["val_hr"] = metrics[hr_key]
        logs["val_ndcg"] = metrics[ndcg_key]
        logger.info(
            "Epoch %3d | Val %s=%.4f | Val %s=%.4f",
            epoch + 1, hr_key, metrics[hr_key], ndcg_key, metrics[ndcg_key],
        )


class RecommenderTrainer:
    """Thin wrapper around `model.compile()` / `model.fit()` for implicit-feedback models.

    Trains with Binary Cross-Entropy loss and Adam optimisation using Keras's
    built-in training loop. Validation HR@K/NDCG@K is computed each epoch by
    `_RankingEvalCallback`; early stopping and best-weight restoration are
    handled by `tf.keras.callbacks.EarlyStopping` (replaces the manual
    state-dict cloning from the PyTorch version).

    Args:
        model: A `tf.keras.Model` with a `call((user, item), training) -> Tensor` signature.
        epochs: Maximum number of training epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size for the training dataset.
        patience: Early-stopping tolerance in validation epochs.
        eval_k: Cutoff rank K passed to `RankingEvaluator`.
        eval_batch_size: Users scored per forward pass during validation.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 1024,
        patience: int = 5,
        eval_k: int = 10,
        eval_batch_size: int = 256,
    ) -> None:
        self.model: tf.keras.Model = model
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.patience: int = patience
        self._evaluator = RankingEvaluator(k=eval_k, eval_batch_size=eval_batch_size)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
        )

    def fit(self, train_dataset, val_dataset) -> Tuple[tf.keras.Model, Dict[str, List[float]]]:
        """Run `model.fit()` with ranking-eval and early-stopping callbacks.

        Args:
            train_dataset: `TrainDataset` (TF version, from `data.py`);
                `.as_tf_dataset()` is called internally with `self.batch_size`.
            val_dataset: `EvalDataset` (TF version, from `data.py`) used for
                validation ranking scoring.

        Returns:
            Tuple of (best_model, history_dict) where history_dict contains
            lists "train_loss", "val_hr", and "val_ndcg" — best weights
            (by val_hr) are already restored onto `best_model` via
            `EarlyStopping(restore_best_weights=True)`.

        Raises:
            RuntimeError: If `model.fit()` raises an unrecoverable error.
        """
        train_ds = train_dataset.as_tf_dataset(batch_size=self.batch_size, shuffle=True)

        eval_cb = _RankingEvalCallback(self._evaluator, val_dataset)
        early_stop_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_hr",
            mode="max",
            patience=self.patience,
            restore_best_weights=True,
            verbose=1,
        )

        try:
            fit_result = self.model.fit(
                train_ds,
                epochs=self.epochs,
                callbacks=[eval_cb, early_stop_cb],
                verbose=1,
            )
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            raise RuntimeError("Training aborted") from exc

        history: Dict[str, List[float]] = {
            "train_loss": fit_result.history.get("loss", []),
            "val_hr": fit_result.history.get("val_hr", []),
            "val_ndcg": fit_result.history.get("val_ndcg", []),
        }
        return self.model, history
