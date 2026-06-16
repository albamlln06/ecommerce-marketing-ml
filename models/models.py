import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from cornac.models import BPR, MF, MostPop


# ─────────────────────────────────────────────
# CONTENT-BASED
# ─────────────────────────────────────────────

class ContentBasedRecommender:
    """
    Cosine similarity over the product feature matrix.
    Works with a single interaction — good for cold-start on Olist.
    """

    def __init__(self, product_matrix, product_ids):
        self.product_matrix = product_matrix
        self.product_ids    = product_ids
        self.id_to_idx      = {pid: i for i, pid in enumerate(product_ids)}

    def fit(self):
        print("Computing cosine similarity matrix...")
        self.sim_matrix = cosine_similarity(self.product_matrix)
        np.fill_diagonal(self.sim_matrix, -1)
        print(f"Similarity matrix: {self.sim_matrix.shape}")
        return self

    def recommend(self, product_ids_seen, top_k=10, exclude_seen=True):
        seen_indices = [
            self.id_to_idx[pid]
            for pid in product_ids_seen
            if pid in self.id_to_idx
        ]

        if not seen_indices:
            scores = self.product_matrix[:, -1].copy()
        else:
            scores = self.sim_matrix[seen_indices].mean(axis=0).copy()

        if exclude_seen:
            for idx in seen_indices:
                scores[idx] = -1

        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.product_ids[i], float(scores[i])) for i in top_indices]


def train_content_based(data):
    print("=" * 55)
    print("Training Content-Based Recommender...")
    print("=" * 55)
    model = ContentBasedRecommender(data["product_matrix"], data["product_ids"])
    model.fit()
    print("✓ Content-Based ready.\n")
    return model


# ─────────────────────────────────────────────
# CORNAC — BPR
# ─────────────────────────────────────────────

def train_bpr(data, k=50, max_iter=200, learning_rate=0.01, seed=42):
    """
    BPR — Bayesian Personalized Ranking.
    Optimizes ranking directly. Best option for implicit feedback (purchases).
    """
    print("=" * 55)
    print("Training BPR (Bayesian Personalized Ranking)...")
    print("=" * 55)
    model = BPR(
        k=k,
        max_iter=max_iter,
        learning_rate=learning_rate,
        lambda_reg=0.001,
        seed=seed,
        verbose=True,
    )
    model.fit(data["dataset_cornac"])
    print("✓ BPR ready.\n")
    return model


# ─────────────────────────────────────────────
# CORNAC — MF
# ─────────────────────────────────────────────

def train_mf(data, k=50, max_iter=200, learning_rate=0.01, seed=42):
    """
    MF — Matrix Factorization with SGD.
    Classic collaborative filtering baseline to compare against BPR.
    """
    print("=" * 55)
    print("Training MF (Matrix Factorization)...")
    print("=" * 55)
    model = MF(
        k=k,
        max_iter=max_iter,
        learning_rate=learning_rate,
        lambda_reg=0.001,
        use_bias=True,
        seed=seed,
        verbose=True,
    )
    model.fit(data["dataset_cornac"])
    print("✓ MF ready.\n")
    return model


# ─────────────────────────────────────────────
# CORNAC — MOST POPULAR (baseline)
# ─────────────────────────────────────────────

def train_most_popular(data):
    """
    Most Popular — recommends the most purchased products.
    Strong baseline in sparse datasets. CF models must beat this.
    """
    print("=" * 55)
    print("Training Most Popular (baseline)...")
    print("=" * 55)
    model = MostPop()
    model.fit(data["dataset_cornac"])
    print("✓ Most Popular ready.\n")
    return model


# ─────────────────────────────────────────────
# RANDOM (minimum baseline)
# ─────────────────────────────────────────────

class RandomRecommender:
    """Lower bound: any model must outperform random recommendations."""

    def __init__(self, product_ids, seed=42):
        self.product_ids = product_ids
        self.rng         = np.random.default_rng(seed)

    def fit(self, *args, **kwargs):
        return self

    def recommend(self, user_id, top_k=10, exclude_seen=None):
        exclude_seen = set(exclude_seen or [])
        candidates   = [p for p in self.product_ids if p not in exclude_seen]
        chosen       = self.rng.choice(candidates, size=min(top_k, len(candidates)), replace=False)
        return [(pid, 0.0) for pid in chosen]


def train_random(data, seed=42):
    print("=" * 55)
    print("Training Random (minimum baseline)...")
    print("=" * 55)
    model = RandomRecommender(data["product_ids"], seed=seed)
    print("✓ Random ready.\n")
    return model


# ─────────────────────────────────────────────
# SHARED CORNAC RECOMMENDATION HELPER
# ─────────────────────────────────────────────

def recommend_for_user_cornac(customer_id, model, dataset_cornac, top_n=10):
    """
    Returns (product_id, score) list from any Cornac model.
    Raises ValueError if the user was not seen during training.
    """
    if customer_id not in dataset_cornac.uid_map:
        raise ValueError(f"Customer {customer_id} not in training set")
    # Cornac 2.5+ returns a list of item_id strings (not tuples) and requires
    # train_set when remove_seen=True
    item_ids = model.recommend(user_id=customer_id, k=top_n, remove_seen=True, train_set=dataset_cornac)
    return [(iid, 0.0) for iid in item_ids]


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def train_all(data, k=50, max_iter=200, learning_rate=0.01, seed=42):
    """Train all models and return them in a dict keyed by name."""
    models = {
        "content_based": train_content_based(data),
        "bpr":           train_bpr(data, k=k, max_iter=max_iter, learning_rate=learning_rate, seed=seed),
        "mf":            train_mf(data,  k=k, max_iter=max_iter, learning_rate=learning_rate, seed=seed),
        "most_popular":  train_most_popular(data),
        "random":        train_random(data, seed=seed),
    }
    print("=" * 55)
    print("✓ All models trained:", list(models.keys()))
    print("=" * 55)
    return models
