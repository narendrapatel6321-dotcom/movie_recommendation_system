# Neural Collaborative Filtering for Implicit-Feedback Recommendation

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Dataset](https://img.shields.io/badge/Dataset-MovieLens%201M-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Matrix Factorization vs. MLP vs. NeuMF — a controlled architecture comparison for top-N movie recommendation, evaluated under a leave-one-out ranking protocol on MovieLens 1M.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repo Structure](#repo-structure)
- [Dataset](#dataset)
- [Recommendation Architecture](#recommendation-architecture)
- [Data Engineering & Preprocessing](#data-engineering--preprocessing)
- [Similarity Metrics & Ranking](#similarity-metrics--ranking)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Visualizations](#visualizations)
- [Inference](#inference)
- [Reproducibility](#reproducibility)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [License](#license)

---

## Project Overview

**Why.** In a B2C streaming or media product, the cost of a poor recommendation is not a wrong classification label — it's a user who scrolls past the entire row and closes the app. Ranking quality at the top of the list (top-10, not top-1000) is what drives session-level engagement, so any recommender needs to be evaluated the way it will actually be consumed: as a ranking problem over a small candidate set, not as a regression or classification task over the full catalogue.

**What.** This project implements and benchmarks three neural collaborative filtering (NCF) architectures of increasing representational capacity on MovieLens 1M (1M ratings, 6,040 users, 3,706 movies):

1. **Matrix Factorization (MF)** — bilinear dot-product interaction with bias terms.
2. **Multi-Layer Perceptron (MLP)** — learned nonlinear interaction over concatenated embeddings.
3. **Neural Matrix Factorization (NeuMF)** — a fused two-tower model combining a GMF (generalized matrix factorization) branch with an MLP branch (He et al., 2017).

All three are trained and evaluated under **identical data splits, negative-sampling protocol, and evaluation metrics**, so that differences in HR@10 / NDCG@10 can be attributed to model architecture rather than experimental setup. Model-specific hyperparameters (embedding size, learning rate, epoch budget) are still tuned independently, since forcing identical hyperparameters across architectures of different capacity would bias the comparison against the simpler model.

| Model | Interaction Function | Parameters (est.)* | Capacity |
|---|---|---|---|
| MF | `sigmoid(p_u · q_i + b_u + b_i)` | ~0.63M | Linear (bilinear) |
| MLP | `sigmoid(MLP([p_u ‖ q_i]))` | ~0.32M | Nonlinear, single tower |
| NeuMF | `sigmoid(W · [GMF(p_u,q_i) ‖ MLP([p_u ‖ q_i])])` | ~0.63M | Nonlinear, fused two-tower |

*Estimated from standard ML-1M dimensions (6,040 users × 3,706 items) at the embedding sizes in `COMMON_CONFIG`/`*_CONFIG`; exact counts are printed by `model.summary()` at build time.

---

## Repo Structure

```
movie_recommendation_system/
├── neural_recommender_system.py   # Colab-exported driver script (MF → MLP → NeuMF)
├── requirements.txt
├── data/
│   └── ml-1m/                     # ratings.dat, movies.dat, users.dat (downloaded at runtime)
├── results/                       # saved weights, training_history.png, model_comparison.png
└── src/
    ├── __init__.py                # exposes MF, MLP, NeuMF
    ├── data.py                    # ingestion, ID remapping, leave-one-out split, tf.data pipelines
    ├── models.py                  # MatrixFactorization, MLP, NeuMF (tf.keras.Model subclasses)
    ├── train.py                   # RecommenderTrainer: compile/fit + ranking-aware early stopping
    ├── evaluate.py                # RankingEvaluator: HR@K, NDCG@K
    └── utils.py                   # seeding, checkpointing, plotting
```

## Trained Models

Checkpoints are saved as Keras weights-only files (`*.weights.h5`, via `save_weights`) rather than full `SavedModel` exports, since all three models are subclassed (not Functional/Sequential) and don't carry a registered `get_config()`/`from_config()`. To reload a checkpoint, the model must first be built with an identical architecture (a single forward pass on dummy tensors), then `load_weights` matches parameters by layer structure. Checkpoints live in `results/{mf,mlp,neumf}.weights.h5`.

---

## Dataset

**MovieLens 1M**: 1,000,209 ratings from 6,040 users on 3,706 movies (files.grouplens.org). Only implicit-feedback signal (interaction / no interaction) is used — explicit star ratings are not modeled.

**ID remapping.** Raw `user_id` / `movie_id` values in `ratings.dat` are non-contiguous. `MovieLensDataLoader._remap_ids` builds a sorted-unique-ID → zero-based-index mapping for both users and items, which is required for direct use as `tf.keras.layers.Embedding` indices (embedding tables are dense lookup arrays indexed `[0, n)`).

**Leave-one-out temporal split.** Per user, interactions are ranked by timestamp (`groupby("user")["timestamp"].rank(method="first", ascending=False)`):

- `rank == 1` (most recent) → **test**
- `rank == 2` (second most recent) → **validation**
- `rank > 2` (everything earlier) → **train**

This is the standard NCF evaluation protocol (He et al., 2017): it simulates the real deployment scenario of predicting a user's *next* interaction from their history, rather than a random split that would leak future interactions into training.

---

## Recommendation Architecture

This is a **pure collaborative filtering** system — no content/metadata features (genres, cast, tags, TF-IDF over synopses) are used. Each user and item is represented solely by a learned latent embedding vector, fit purely from the interaction matrix. This is a deliberate scope choice: it isolates the effect of interaction-modeling architecture (bilinear vs. nonlinear vs. fused) without confounding from feature engineering, which is the variable under test in this comparison.

### MF (baseline)
```
score(u, i) = sigmoid( p_u · q_i + b_u + b_i )
```
`p_u`, `q_i` are learned embedding vectors (`embed_dim=64`); `b_u`, `b_i` are learned scalar bias terms. This is the classical bilinear MF formulation, generalized to implicit feedback via a sigmoid link and binary cross-entropy loss (rather than squared-error reconstruction of explicit ratings).

### MLP
```
x = [p_u ‖ q_i]                       # concatenation, not dot product
score(u, i) = sigmoid( Dense(1)( MLPTower(x) ) )
```
User and item embeddings (`embed_dim=32` each, concatenated to 64-d) are passed through a fully connected tower (`Dense(64) → Dropout(0.2) → Dense(32) → Dropout(0.2) → Dense(16) → Dropout(0.2)`) with ReLU activations, replacing the fixed bilinear form with a learned, arbitrarily nonlinear interaction function.

### NeuMF
```
gmf_out = p_u_mf ⊙ q_i_mf                          # element-wise product, mf_dim=32
mlp_out = MLPTower([p_u_mlp ‖ q_i_mlp])             # separate embedding table, mlp_embed=32
score(u, i) = sigmoid( Dense(1)( [gmf_out ‖ mlp_out] ) )
```
NeuMF maintains **two independent embedding tables per user/item** (one for the GMF branch, one for the MLP branch) rather than sharing a single embedding space — this lets each branch specialize its representation to the interaction function it feeds, which is the key architectural insight from He et al. (2017) over naively summing MF and MLP scores.

---

## Data Engineering & Preprocessing

- **No missing-value imputation is needed**: implicit-feedback data has no null ratings by construction — a (user, item) pair either appears in the interaction log or it doesn't.
- **No categorical metadata** (genres, cast, tags) is incorporated; the feature space is entirely learned embeddings, so there is no TF-IDF/`CountVectorizer`/one-hot encoding step in this pipeline.
- **Negative sampling (training).** For every observed positive interaction, `TrainDataset` draws `num_negatives=4` items uniformly at random from the full catalogue, excluding items already in that user's positive set (`user_pos_items`). Sampling uses a seeded `np.random.default_rng` and a bounded retry loop (`max_attempts = n_items * 10`) with a warning if a user's positive set is dense enough that negative sampling starts stalling.
- **Negative sampling (evaluation).** `EvalDataset` builds a fixed candidate set per user: the one held-out positive item at index 0, plus `eval_negatives=99` sampled negatives — the standard 1-vs-99 ranking protocol, decorrelated from training negatives via `seed + 1`.
- **Sparse interaction matrix.** The user–item matrix is never materialized densely or as a `scipy.sparse` matrix; it is implicit in the embedding-lookup formulation — the model only ever touches the small batch of (user, item) index pairs sampled per step, which is what makes this approach scale to the full ML-1M catalogue without an `O(n_users × n_items)` memory footprint.
- **Pipeline.** `tf.data.Dataset.from_tensor_slices` → `.shuffle(buffer_size=min(N, 10_000), seed=...)` → `.batch()` → `.prefetch(AUTOTUNE)`, yielding `((user, item), label)` tuples matching Keras's `model.fit(x, y)` calling convention directly.

---

## Similarity Metrics & Ranking

No explicit similarity metric (cosine, Pearson, Euclidean) is computed post hoc — similarity is implicit in each model's own interaction function (dot product for MF's GMF-style scoring, learned nonlinear function for MLP/NeuMF). All three models output a single scalar **interaction probability** `ŷ_ui ∈ (0, 1)` per (user, item) pair via a final sigmoid.

**Ranking / evaluation protocol (`RankingEvaluator`):**

1. For each test user, score all 100 candidates (1 positive + 99 sampled negatives) in a single batched forward pass.
2. Rank candidates by predicted score; compute the 0-indexed rank of the positive item as `(scores > pos_score).sum()`.
3. **Hit Rate@10 (HR@10):** `1` if the positive item's rank `< 10`, else `0`, averaged over all users. Measures whether the model surfaces the relevant item anywhere in the top-10 slate — the recommendation-row-level engagement metric.
4. **NDCG@10:** `1 / log2(rank + 2)` if rank `< 10`, else `0`. Unlike HR@10, this rewards placing the relevant item *higher* within the top-10 — directly relevant to a UI where row position affects click-through.

This is a **fixed-candidate ranking evaluation**, not full-catalogue retrieval — a deliberate, standard simplification (He et al., 2017) that keeps evaluation compute tractable while still measuring true ranking quality rather than pointwise classification accuracy.

---

## Training Configuration

| Setting | MF | MLP | NeuMF |
|---|---|---|---|
| Embedding dim | 64 | 32 (×2 concat) | 32 (GMF) + 32 (MLP, ×2 concat) |
| Hidden layers | — | (64, 32, 16) | (64, 32, 16) |
| Dropout | — | 0.2 | 0.2 |
| Learning rate | 1e-3 | 1e-3 | 5e-4 |
| Max epochs | 30 | 50 | 30 |
| Batch size | 4,096 | 4,096 | 4,096 |
| Negatives (train / eval) | 4 / 99 | 4 / 99 | 4 / 99 |
| Optimizer / Loss | Adam / Binary Cross-Entropy | Adam / BCE | Adam / BCE |
| Early stopping | `val_hr`, patience 5, best weights restored | same | same |

**Monitor choice.** Early stopping and best-weight restoration are keyed on **validation HR@10**, not validation loss — BCE loss on uniformly sampled negatives does not track ranking quality closely enough to be a reliable stopping signal for a ranking task; a `_RankingEvalCallback` runs the full `RankingEvaluator` pass at the end of every epoch and injects `val_hr`/`val_ndcg` into the Keras `logs` dict so `EarlyStopping` can act on it as a native metric.

---

## Results

> **[TODO: populate after the training run completes — the source notebook has `HR@10: ...` / `NDCG@10: ...` placeholders in the MF interpretation cell.]**

| Model | HR@10 | NDCG@10 |
|---|---|---|
| MF | *TBD* | *TBD* |
| MLP | *TBD* | *TBD* |
| NeuMF | *TBD* | *TBD* |

> Best model: *[fill in once results are in — update the bold row and this callout together]*

### Key Takeaways
- *[TODO — e.g., whether added architectural capacity (MLP, NeuMF) translated into ranking gains over the MF baseline, or whether MF's stronger inductive bias for this dataset/training budget won out, as is common in NCF benchmarks at this scale.]*

### Failed Experiments
- *[TODO — log any configs that underperformed or didn't converge, per your standard portfolio format.]*

### Limitations
- Pure collaborative filtering: cold-start users/items with no interaction history cannot be scored — a content-based or hybrid fallback would be required in production.
- Fixed-candidate (1-vs-99) evaluation approximates full-catalogue ranking but is not identical to it; HR@10/NDCG@10 under this protocol are not directly comparable to full-corpus retrieval metrics.
- Negative sampling is uniform-random rather than popularity-aware, which can under-penalize models that simply favor popular items.

---

## Visualizations

`plot_training_history` (in `utils.py`) renders a 3-panel figure — training loss, validation HR@K, validation NDCG@K — across all three models on shared axes, saved to `results/training_history.png`. `plot_comparison_bar` renders a grouped bar chart of final test-set HR@10/NDCG@10 per model, saved to `results/model_comparison.png`.

*(Insert `results/training_history.png` and `results/model_comparison.png` here once generated.)*

---

## Inference

To score a candidate slate for a given user with a trained model:

```python
import tensorflow as tf
from src.models import create_model
from src.utils import load_model_weights

model = create_model("NeuMF", n_users=6040, n_items=3706, config=NEUMF_CONFIG)
_ = model((tf.zeros((1,), dtype=tf.int64), tf.zeros((1,), dtype=tf.int64)))  # build
model = load_model_weights(model, name="neumf", save_dir="results")

user_id = tf.constant([42] * len(candidate_item_ids), dtype=tf.int64)
item_ids = tf.constant(candidate_item_ids, dtype=tf.int64)
scores = model((user_id, item_ids), training=False)
top_n = tf.argsort(scores, direction="DESCENDING")[:10]
```

---

## Reproducibility

`set_seed(seed)` fixes Python's `random`, NumPy, and `tf.random` seeds. `COMMON_CONFIG["seed"] = 42` propagates through negative sampling in both `TrainDataset` (seed) and `EvalDataset` (`seed + 1`, keeping eval negatives decorrelated from training negatives) so that dataset construction is deterministic across runs. Model weight initialization uses `RandomNormal(stddev=0.01)` for all embedding tables.

---

## Tech Stack

- **TensorFlow / Keras** — model definitions (`tf.keras.Model` subclasses), `model.fit()` training loop, custom callbacks
- **Pandas** — ratings ingestion, ID remapping, leave-one-out split logic
- **NumPy** — negative-sampling RNG, ranking/metric computation in `RankingEvaluator`
- **Matplotlib** — training-history and model-comparison plots
- **tf.data** — batching, shuffling, prefetch pipelines

---

## How to Run

```bash
git clone https://github.com/narendrapatel6321-dotcom/movie_recommendation_system.git
cd movie_recommendation_system
pip install -q -r requirements.txt
```

The MovieLens 1M archive is downloaded and extracted automatically on first run if `data/ml-1m/ratings.dat` is not found. Then, either run the exported driver script end-to-end:

```bash
python neural_recommender_system.py
```

or, in a notebook/Colab environment, execute the cell blocks in `neural_recommender_system.py` sequentially (data load → MF → MLP → NeuMF → final comparison), which is the intended workflow given the `!git clone` / `%cd` magics at the top of the file.

---

## License

MIT
