"""MovieLens 1M ingestion, ID remapping, and leave-one-out splitting."""

import logging
import os
from typing import Dict, Set

import pandas as pd

logger = logging.getLogger(__name__)


class MovieLensDataLoader:
    """Loads and partitions the MovieLens 1M rating dataset.

    Applies contiguous ID remapping so user and item indices are zero-based
    integers suitable for embedding layers.  Uses a leave-one-out temporal
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
