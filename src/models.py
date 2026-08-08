"""Matrix Factorization, MLP, and NeuMF models for implicit-feedback recommendation."""

from typing import Any, Mapping, Tuple

import tensorflow as tf


class MatrixFactorization(tf.keras.Model):
    """Classical dot-product Matrix Factorization with per-user and per-item biases.

    Scores a (user, item) pair as:
        sigmoid( <p_u, q_i> + b_u + b_i )

    where p_u and q_i are learned embedding vectors and b_u, b_i are scalar
    bias terms. Output is passed through sigmoid so binary cross-entropy loss
    can be applied directly for implicit-feedback training.

    Args:
        n_users: Total number of users in the dataset.
        n_items: Total number of items in the dataset.
        embed_dim: Dimensionality of user and item embedding vectors.
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64, **kwargs) -> None:
        super().__init__(**kwargs)
        emb_init = tf.keras.initializers.RandomNormal(stddev=0.01)

        self.user_emb = tf.keras.layers.Embedding(
            n_users, embed_dim, embeddings_initializer=emb_init, name="user_emb"
        )
        self.item_emb = tf.keras.layers.Embedding(
            n_items, embed_dim, embeddings_initializer=emb_init, name="item_emb"
        )
        self.user_bias = tf.keras.layers.Embedding(
            n_users, 1, embeddings_initializer="zeros", name="user_bias"
        )
        self.item_bias = tf.keras.layers.Embedding(
            n_items, 1, embeddings_initializer="zeros", name="item_bias"
        )

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """Compute sigmoid-normalised interaction scores for a batch.

        Args:
            inputs: Tuple of (user, item) int tensors, each shape (B,). This
                tuple form is required so `model.fit()` can call the model
                directly on batches yielded as `((user, item), label)`.
            training: Unused (no training-only ops in this model); kept for API parity.

        Returns:
            Float tensor of predicted interaction probabilities, shape (B,).
        """
        user, item = inputs
        pu = self.user_emb(user)
        qi = self.item_emb(item)
        bu = tf.squeeze(self.user_bias(user), axis=-1)
        bi = tf.squeeze(self.item_bias(item), axis=-1)
        dot = tf.reduce_sum(pu * qi, axis=-1)
        return tf.sigmoid(dot + bu + bi)


class MLP(tf.keras.Model):
    """NCF-MLP: concatenated user/item embeddings fed through a fully-connected tower.

    The user embedding p_u and item embedding q_i are concatenated into a single
    vector [p_u || q_i], which is then passed through a stack of dense→ReLU→dropout
    layers. A final linear projection maps to a scalar interaction score normalised
    by sigmoid.

    Args:
        n_users: Total number of users.
        n_items: Total number of items.
        embed_dim: Dimensionality of each user/item embedding (concatenated input = 2×embed_dim).
        layers: Sequence of hidden-layer widths for the MLP tower.
        dropout: Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 32,
        layers: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        emb_init = tf.keras.initializers.RandomNormal(stddev=0.01)

        self.user_emb = tf.keras.layers.Embedding(
            n_users, embed_dim, embeddings_initializer=emb_init, name="user_emb"
        )
        self.item_emb = tf.keras.layers.Embedding(
            n_items, embed_dim, embeddings_initializer=emb_init, name="item_emb"
        )

        # Dense's default kernel_initializer is already Glorot/Xavier uniform, so —
        # unlike the PyTorch version — no output-layer init fix is needed here.
        self.hidden_layers = []
        for out_dim in layers:
            self.hidden_layers.append(tf.keras.layers.Dense(out_dim, activation="relu"))
            self.hidden_layers.append(tf.keras.layers.Dropout(dropout))
        self.output_layer = tf.keras.layers.Dense(1)

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """Compute sigmoid-normalised interaction scores for a batch.

        Args:
            inputs: Tuple of (user, item) int tensors, each shape (B,). This
                tuple form is required so `model.fit()` can call the model
                directly on batches yielded as `((user, item), label)`.
            training: Whether dropout should be active.

        Returns:
            Float tensor of predicted interaction probabilities, shape (B,).
        """
        user, item = inputs
        x = tf.concat([self.user_emb(user), self.item_emb(item)], axis=-1)
        for layer in self.hidden_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return tf.sigmoid(tf.squeeze(self.output_layer(x), axis=-1))


class NeuMF(tf.keras.Model):
    """Neural Matrix Factorization: fused GMF and MLP branches (He et al. 2017).

    GMF branch:
        p_u_mf ⊙ q_i_mf → element-wise product vector of shape (mf_dim,)

    MLP branch:
        [p_u_mlp || q_i_mlp] → configurable fully-connected tower

    Fusion:
        [gmf_out || mlp_out] → Dense(1) → sigmoid

    Separate embedding matrices per branch allow each path to learn a
    complementary representation: GMF captures bilinear interactions while
    MLP captures non-linear patterns.

    Args:
        n_users: Total number of users in the dataset.
        n_items: Total number of items in the dataset.
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        emb_init = tf.keras.initializers.RandomNormal(stddev=0.01)

        self.mf_user = tf.keras.layers.Embedding(
            n_users,
            mf_dim,
            embeddings_initializer=emb_init,
            name="mf_user",
        )
        self.mf_item = tf.keras.layers.Embedding(
            n_items,
            mf_dim,
            embeddings_initializer=emb_init,
            name="mf_item",
        )
        self.mlp_user = tf.keras.layers.Embedding(
            n_users,
            mlp_embed,
            embeddings_initializer=emb_init,
            name="mlp_user",
        )
        self.mlp_item = tf.keras.layers.Embedding(
            n_items,
            mlp_embed,
            embeddings_initializer=emb_init,
            name="mlp_item",
        )

        # Dense's default kernel_initializer is already Glorot/Xavier uniform, so —
        # unlike the PyTorch version — no output-layer init fix is needed here.
        self.hidden_layers = []
        for out_dim in layers:
            self.hidden_layers.append(tf.keras.layers.Dense(out_dim, activation="relu"))
            self.hidden_layers.append(tf.keras.layers.Dropout(dropout))
        self.output_layer = tf.keras.layers.Dense(1)

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """Compute fused GMF+MLP interaction scores for a batch.

        Args:
            inputs: Tuple of (user, item) int tensors, each shape (B,). This
                tuple form is required so `model.fit()` can call the model
                directly on batches yielded as `((user, item), label)`.
            training: Whether dropout should be active.

        Returns:
            Float tensor of predicted interaction probabilities, shape (B,).
        """
        user, item = inputs
        gmf_out = self.mf_user(user) * self.mf_item(item)

        x = tf.concat([self.mlp_user(user), self.mlp_item(item)], axis=-1)
        for layer in self.hidden_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)

        fused = tf.concat([gmf_out, x], axis=-1)
        return tf.sigmoid(tf.squeeze(self.output_layer(fused), axis=-1))


def create_model(
    model_name: str,
    n_users: int,
    n_items: int,
    config: Mapping[str, Any],
) -> tf.keras.Model:
    """Construct one of the supported recommender architectures.

    Args:
        model_name: Model identifier: ``"MF"``, ``"MLP"``, or ``"NeuMF"``.
        n_users: Total number of users in the dataset.
        n_items: Total number of items in the dataset.
        config: Model-specific architecture configuration.

    Returns:
        An unbuilt TensorFlow recommender model.

    Raises:
        ValueError: If ``model_name`` is not supported.
    """
    if model_name == "MF":
        return MatrixFactorization(
            n_users,
            n_items,
            embed_dim=config["embed_dim"],
        )

    if model_name == "MLP":
        return MLP(
            n_users,
            n_items,
            embed_dim=config["embed_dim"],
            layers=config["layers"],
            dropout=config["dropout"],
        )

    if model_name == "NeuMF":
        return NeuMF(
            n_users,
            n_items,
            mf_dim=config["mf_dim"],
            mlp_embed=config["mlp_embed"],
            layers=config["layers"],
            dropout=config["dropout"],
        )

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        "Expected one of: 'MF', 'MLP', 'NeuMF'."
    )
