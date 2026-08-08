"""Plotting helpers, model persistence, and reproducibility utilities."""

import logging
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow for reproducible runs.

    Args:
        seed: Integer seed value to propagate to all RNG backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.debug("Global seed set to %d", seed)


def save_model(model: tf.keras.Model, name: str, save_dir: str = "results") -> None:
    """Persist a model's weights to disk as a Keras ``.weights.h5`` checkpoint.

    Uses ``save_weights`` (not the full ``model.save``) since these models
    are subclassed rather than functional/Sequential — weights-only
    checkpoints avoid needing a registered ``get_config``/``from_config``
    for full-model serialisation.

    Args:
        model: The trained ``tf.keras.Model`` whose weights should be saved.
        name: Base filename (without extension) for the checkpoint.
        save_dir: Directory in which to write the file (created if absent).

    Raises:
        OSError: If the directory cannot be created or the file cannot be written.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.weights.h5")
        model.save_weights(path)
        logger.info("Model checkpoint saved to '%s'", path)
    except OSError as exc:
        logger.error("Failed to save model '%s': %s", name, exc)
        raise


def load_model_weights(model: tf.keras.Model, name: str, save_dir: str = "results") -> tf.keras.Model:
    """Load a previously saved checkpoint's weights into ``model`` in place.

    The model must already be built (e.g. by calling it once on a sample
    batch, or via ``model.build(...)``) with the same architecture used to
    produce the checkpoint, since ``save_weights``/``load_weights`` match
    parameters by layer structure rather than storing the architecture itself.

    Args:
        model: A ``tf.keras.Model`` instance with the same architecture as
            the one the checkpoint was saved from.
        name: Base filename (without extension) the checkpoint was saved under.
        save_dir: Directory the checkpoint file lives in.

    Returns:
        The same ``model`` instance, with weights loaded in place.

    Raises:
        OSError: If the checkpoint file cannot be found or read.
    """
    path = os.path.join(save_dir, f"{name}.weights.h5")
    try:
        model.load_weights(path)
        logger.info("Model checkpoint loaded from '%s'", path)
    except OSError as exc:
        logger.error("Failed to load model '%s': %s", name, exc)
        raise
    return model


def plot_training_history(
    histories: Dict[str, Dict[str, List[float]]],
    save_dir: str = "results",
) -> None:
    """Plot BCE training loss and validation HR@10/NDCG@10 curves for all models.

    Args:
        histories: Mapping of model name → history dict with keys
            ``"train_loss"``, ``"val_hr"``, and ``"val_ndcg"``.
        save_dir: Output directory for the PNG file.

    Raises:
        OSError: If the plot file cannot be written.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    try:
        for name, h in histories.items():
            axes[0].plot(h.get("train_loss", []), label=name)
            axes[1].plot(h.get("val_hr", []), label=name)
            axes[2].plot(h.get("val_ndcg", []), label=name)

        configs = [
            ("Training BCE Loss", "Epoch", "Loss"),
            ("Val Hit Rate@10", "Epoch", "HR@10"),
            ("Val NDCG@10", "Epoch", "NDCG@10"),
        ]
        for ax, (title, xlabel, ylabel) in zip(axes, configs):
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()

        plt.tight_layout()
        path = os.path.join(save_dir, "training_history.png")
        plt.savefig(path, dpi=150)
        logger.info("Training history plot saved to '%s'", path)
    except Exception as exc:
        logger.error("Failed to write training history plot: %s", exc)
        raise
    finally:
        plt.close(fig)


def plot_comparison_bar(
    results: Dict[str, Dict[str, float]],
    save_dir: str = "results",
) -> None:
    """Render a grouped bar chart comparing HR@10 and NDCG@10 across models.

    Args:
        results: Mapping of model name → metrics dict (as returned by
            ``RankingEvaluator.evaluate``).
        save_dir: Output directory for the PNG file.

    Raises:
        OSError: If the plot file cannot be written.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_names = list(results.keys())
    metric_names = list(next(iter(results.values())).keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        for i, name in enumerate(model_names):
            vals = [results[name][m] for m in metric_names]
            ax.bar(x + i * width, vals, width, label=name)

        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels([m.upper() for m in metric_names])
        ax.set_ylim(0, 1)
        ax.set_title("Model Comparison on Test Set")
        ax.legend()

        plt.tight_layout()
        path = os.path.join(save_dir, "model_comparison.png")
        plt.savefig(path, dpi=150)
        logger.info("Comparison bar chart saved to '%s'", path)
    except Exception as exc:
        logger.error("Failed to write comparison bar chart: %s", exc)
        raise
    finally:
        plt.close(fig)
