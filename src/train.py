"""Training loop with BCE loss, Adam optimisation, and early stopping."""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .evaluate import RankingEvaluator

logger = logging.getLogger(__name__)


class RecommenderTrainer:
    """Encapsulates the full training loop for implicit-feedback recommendation models.

    Trains with Binary Cross-Entropy loss and Adam optimisation.  At the end of
    every epoch the model is evaluated on the validation set using HR@K; the best
    checkpoint is retained.  Training halts when validation HR@K has not improved
    for ``patience`` consecutive epochs.

    Args:
        model: An ``nn.Module`` with a ``forward(user, item) → Tensor`` signature.
        device: Torch device string (``"cpu"``, ``"cuda"``, or ``"mps"``).
        epochs: Maximum number of training epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size for the training DataLoader.
        patience: Early-stopping tolerance in validation epochs.
        eval_k: Cutoff rank K passed to ``RankingEvaluator``.
        eval_batch_size: Users scored per forward pass during validation/eval.
        num_workers: Worker processes for the training DataLoader.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 1024,
        patience: int = 5,
        eval_k: int = 10,
        eval_batch_size: int = 256,
        num_workers: int = 0,
    ) -> None:
        self.model: nn.Module = model.to(device)
        self.device: str = device
        self.epochs: int = epochs
        self.lr: float = lr
        self.batch_size: int = batch_size
        self.patience: int = patience
        self.num_workers: int = num_workers
        self._evaluator = RankingEvaluator(k=eval_k, eval_batch_size=eval_batch_size)
        self._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._criterion = nn.BCELoss()

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Run the training loop and return the best checkpoint plus history.

        Args:
            train_dataset: ``TrainDataset`` yielding (user, item, label) triples.
            val_dataset: ``EvalDataset`` used for validation HR@K scoring.

        Returns:
            Tuple of (best_model, history_dict) where history_dict contains
            lists ``"train_loss"``, ``"val_hr"``, and ``"val_ndcg"``.

        Raises:
            RuntimeError: If a training step raises an unrecoverable error.
        """
        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        best_hr: float = -1.0
        best_state: Dict = {}
        no_improve: int = 0
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_hr": [],
            "val_ndcg": [],
        }

        hr_key = f"hr@{self._evaluator.k}"
        ndcg_key = f"ndcg@{self._evaluator.k}"

        for epoch in range(1, self.epochs + 1):
            avg_loss = self._train_epoch(loader, epoch)
            val_metrics = self._evaluator.evaluate(self.model, val_dataset, self.device)

            history["train_loss"].append(avg_loss)
            history["val_hr"].append(val_metrics[hr_key])
            history["val_ndcg"].append(val_metrics[ndcg_key])

            logger.info(
                "Epoch %3d/%d | Loss=%.4f | Val %s=%.4f | Val %s=%.4f",
                epoch,
                self.epochs,
                avg_loss,
                hr_key,
                val_metrics[hr_key],
                ndcg_key,
                val_metrics[ndcg_key],
            )

            if val_metrics[hr_key] > best_hr:
                best_hr = val_metrics[hr_key]
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (best %s=%.4f)",
                        epoch,
                        hr_key,
                        best_hr,
                    )
                    break

        self.model.load_state_dict(best_state)
        return self.model, history

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Execute one training epoch and return the mean BCE loss.

        Args:
            loader: DataLoader yielding (user, item, label) batches.
            epoch: Current epoch number (used for tqdm label only).

        Returns:
            Mean per-sample BCE loss over the full epoch.

        Raises:
            RuntimeError: If a forward or backward pass raises an error.
        """
        self.model.train()
        total_loss: float = 0.0
        n_samples: int = 0

        try:
            for user, item, label in tqdm(
                loader, desc=f"Epoch {epoch:3d}/{self.epochs}", leave=False
            ):
                user = user.to(self.device)
                item = item.to(self.device)
                label = label.to(self.device)

                self._optimizer.zero_grad()
                pred: torch.Tensor = self.model(user, item)
                loss: torch.Tensor = self._criterion(pred, label)
                loss.backward()
                self._optimizer.step()

                total_loss += loss.item() * len(user)
                n_samples += len(user)
        except Exception as exc:
            logger.error("Training step failed at epoch %d: %s", epoch, exc)
            raise RuntimeError(f"Training aborted at epoch {epoch}") from exc

        return total_loss / max(n_samples, 1)
