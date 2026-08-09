"""MovieLens 1M ingestion, ID remapping, leave-one-out splitting, and tf.data pipelines."""

import logging
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)


class MovieLensDataLoader:
    """Loads and partitions the MovieLens 1M rating dataset.

    Applies contiguous ID remapping so user and item indices are zero-based
    integers suitable for embedding layers. Uses a leave-one-out temporal
    split: the chronologically last interaction per user is held out as the
    test sample, the second-to-last as validation, and all earlier interactions
    form the training set.

    Args:
        data_dir: Path to the directory containing ``ratings.dat``.

    Raises:
        FileNotFoundError: If ``ratings.dat`` is absent from ``data_dir``.
        ValueError: If the parsed ratings frame contains no rows.
    """

    _RATINGS_FILENAME: str = "ratings.dat"
    _COLUMN_NAMES: tuple = ("user_id", "movie_id", "rating", "timestamp")

    def __init__(self, data_dir: str = "data/ml-1m") -> None:
        self.data_dir: str = data_dir
        self._ratings: pd.DataFrame = pd.DataFrame()
        self.n_users: int = 0
        self.n_items: int = 0
        self.user_pos_items: Dict[int, Set[int]] = {}

    def load(self) -> Dict:
        """Execute the full ingestion and splitting pipeline.

        Returns:
            A dict with keys ``train``, ``val``, ``test`` (each a
            ``pd.DataFrame`` with columns ``user`` and ``item``),
            ``n_users``, ``n_items``, and ``user_pos_items``
            (mapping user index → set of all interacted item indices).

        Raises:
            FileNotFoundError: If the ratings file does not exist.
            ValueError: If the loaded frame is empty after parsing.
        """
        self._read_ratings()
        self._remap_ids()
        self._build_positive_item_sets()
        return self._split()

    def _read_ratings(self) -> None:
        """Parse ratings.dat into a DataFrame, validating existence and content."""
        path = os.path.join(self.data_dir, self._RATINGS_FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Ratings file not found at '{path}'.\n"
                "Download MovieLens 1M from https://grouplens.org/datasets/movielens/1m/ "
                "and place ratings.dat, movies.dat, users.dat in data/ml-1m/."
            )

        logger.info("Reading ratings from '%s'", path)
        try:
            self._ratings = pd.read_csv(
                path,
                sep="::",
                engine="python",
                names=list(self._COLUMN_NAMES),
                encoding="latin-1",
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to parse '{path}': {exc}") from exc

        if self._ratings.empty:
            raise ValueError(f"Ratings file at '{path}' parsed to an empty DataFrame.")

        logger.info(
            "Loaded %d interactions from %d unique users and %d unique items",
            len(self._ratings),
            self._ratings["user_id"].nunique(),
            self._ratings["movie_id"].nunique(),
        )

    def _remap_ids(self) -> None:
        """Remap raw user/movie IDs to contiguous zero-based integer indices."""
        user2idx = {
            uid: i for i, uid in enumerate(sorted(self._ratings["user_id"].unique()))
        }
        movie2idx = {
            mid: i for i, mid in enumerate(sorted(self._ratings["movie_id"].unique()))
        }

        self._ratings["user"] = self._ratings["user_id"].map(user2idx)
        self._ratings["item"] = self._ratings["movie_id"].map(movie2idx)

        self.n_users = len(user2idx)
        self.n_items = len(movie2idx)
        logger.info("Remapped to %d users, %d items", self.n_users, self.n_items)

    def _build_positive_item_sets(self) -> None:
        """Build a per-user set of all observed item indices (used for negative sampling)."""
        self.user_pos_items = self._ratings.groupby("user")["item"].apply(set).to_dict()

    def _split(self) -> Dict:
        """Apply leave-one-out temporal split and return the data manifest."""
        df = self._ratings.sort_values(["user", "timestamp"]).reset_index(drop=True)
        df["rank"] = df.groupby("user")["timestamp"].rank(
            method="first", ascending=False
        )

        train = df[df["rank"] > 2][["user", "item"]].reset_index(drop=True)
        val = df[df["rank"] == 2][["user", "item"]].reset_index(drop=True)
        test = df[df["rank"] == 1][["user", "item"]].reset_index(drop=True)

        logger.info(
            "Split — train: %d | val: %d | test: %d",
            len(train),
            len(val),
            len(test),
        )

        return {
            "train": train,
            "val": val,
            "test": test,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "user_pos_items": self.user_pos_items,
        }


class TrainDataset:
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
        seed: Seed for the negative-sampling RNG.
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
        users, items, labels = self._generate(interactions)
        self.users: np.ndarray = users
        self.items: np.ndarray = items
        self.labels: np.ndarray = labels
        logger.info(
            "TrainDataset: %d samples (%d positives × %d negatives each)",
            len(self.users),
            len(interactions),
            num_negatives,
        )

    def _generate(
        self, interactions: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pre-generate the full arrays of user, item, and label values.

        Args:
            interactions: Positive interaction DataFrame.

        Returns:
            Tuple of (users, items, labels) as int64/int64/float32 arrays.
        """
        users: List[int] = []
        items: List[int] = []
        labels: List[float] = []
        rng = np.random.default_rng(self.seed)
        max_attempts: int = self.n_items * 10

        for row in interactions.itertuples(index=False):
            u, i = int(row.user), int(row.item)
            users.append(u)
            items.append(i)
            labels.append(1.0)

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
                        u,
                        max_attempts,
                        sampled,
                        self.num_negatives,
                    )
                    break
                j = int(rng.integers(self.n_items))
                if j not in pos_set:
                    users.append(u)
                    items.append(j)
                    labels.append(0.0)
                    sampled += 1

        return (
            np.asarray(users, dtype=np.int64),
            np.asarray(items, dtype=np.int64),
            np.asarray(labels, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Return the total number of (user, item, label) triples."""
        return len(self.users)

    def as_tf_dataset(
        self,
        batch_size: int = 1024,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """Wrap the pre-generated arrays in a batched, shuffled ``tf.data.Dataset``.

        Args:
            batch_size: Mini-batch size.
            shuffle: Whether to shuffle each epoch (buffer covers the full dataset).

        Returns:
            A ``tf.data.Dataset`` yielding ``((user, item), label)`` batches —
            the ``(x, y)`` shape ``model.fit()`` expects, where ``x`` is passed
            straight through to the model's ``call(inputs, training)``.
        """
        ds = tf.data.Dataset.from_tensor_slices(
            ((self.users, self.items), self.labels)
        )
        if shuffle:
            shuffle_buffer_size = min(len(self.users), 10_000)

            ds = ds.shuffle(
                buffer_size=shuffle_buffer_size,
                seed=self.seed,
                reshuffle_each_iteration=True,
                )
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


class EvalDataset:
    """Per-user evaluation set following the leave-one-out ranking protocol.

    Each sample contains one positive item (at index 0) followed by
    ``num_negatives`` randomly drawn unobserved items. Callers score all
    candidates and report HR@K / NDCG@K.

    Args:
        pos_interactions: DataFrame with the held-out positive (user, item) pairs.
        user_pos_items: Full set of each user's observed items (for exclusion).
        n_items: Total catalogue size.
        num_negatives: Number of negative candidates per evaluation sample (default 99).
        seed: Seed for the negative-sampling RNG.
    """

    def __init__(
        self,
        pos_interactions: pd.DataFrame,
        user_pos_items: Dict[int, Set[int]],
        n_items: int,
        num_negatives: int = 99,
        seed: int = 0,
    ) -> None:
        self.seed: int = seed
        users: List[int] = []
        candidates: List[List[int]] = []
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
                        u,
                        max_attempts,
                        len(negs),
                        num_negatives,
                    )
                    break
                j = int(rng.integers(n_items))
                if j not in pos_set and j not in negs:
                    negs.append(j)

            users.append(u)
            candidates.append([pos_i] + negs)

        self.users: np.ndarray = np.asarray(users, dtype=np.int64)
        self.candidates: np.ndarray = np.asarray(candidates, dtype=np.int64)

        logger.info(
            "EvalDataset: %d users × %d candidates each",
            len(self.users),
            1 + num_negatives,
        )

    def __len__(self) -> int:
        """Return the number of evaluation users."""
        return len(self.users)

    def as_tf_dataset(self, batch_size: int = 256) -> tf.data.Dataset:
        """Wrap the pre-generated arrays in a batched, shuffled ``tf.data.Dataset``.

        Args:
            batch_size: Number of users scored per forward pass.

        Returns:
            A ``tf.data.Dataset`` yielding ``(user, candidates)`` batches, where
            ``candidates`` has shape ``(batch, 1 + num_negatives)`` and the
            positive item is always at column 0.
        """
        ds = tf.data.Dataset.from_tensor_slices((self.users, self.candidates))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


def create_datasets(
    data: Dict,
    num_negatives: int = 4,
    eval_negatives: int = 99,
    seed: int = 42,
) -> Tuple[TrainDataset, EvalDataset, EvalDataset]:
    """Create training, validation, and test datasets from a loaded data manifest.

    Args:
        data: Data manifest returned by ``MovieLensDataLoader.load()``.
        num_negatives: Number of negative samples per positive training
            interaction.
        eval_negatives: Number of negative candidates per validation/test
            interaction.
        seed: Seed used for training negative sampling. Evaluation datasets
            use ``seed + 1`` to keep their negative samples decorrelated from
            training negatives.

    Returns:
        Tuple of ``(train_dataset, val_dataset, test_dataset)``.
    """
    n_items = data["n_items"]
    pos_items = data["user_pos_items"]

    train_dataset = TrainDataset(
        data["train"],
        n_items,
        pos_items,
        num_negatives=num_negatives,
        seed=seed,
    )

    val_dataset = EvalDataset(
        data["val"],
        pos_items,
        n_items,
        num_negatives=eval_negatives,
        seed=seed + 1,
    )

    test_dataset = EvalDataset(
        data["test"],
        pos_items,
        n_items,
        num_negatives=eval_negatives,
        seed=seed + 1,
    )

    return train_dataset, val_dataset, test_dataset
