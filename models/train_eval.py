import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Allow running as: python models/train_eval.py  OR  python -m models.train_eval
_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)       # for src.utils.data_preprocess
sys.path.insert(0, _MODEL_DIR)  # for models.py (same directory)

from src.utils.data_preprocess import prepare_all
import models as m  # imports models/models.py


# ─────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────

def get_test_relevant_items(data):
    """Returns {customer_id: set(product_ids)} for users present in both train and test."""
    test        = data["df_test"]
    train_users = set(data["df_train"]["customer_unique_id"])
    return (
        test[test["customer_unique_id"].isin(train_users)]
        .groupby("customer_unique_id")["product_id"]
        .apply(set)
        .to_dict()
    )


def precision_recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    hits          = len(set(recommended_k) & relevant)
    precision     = hits / k
    recall        = hits / len(relevant) if relevant else 0.0
    return precision, recall


def ndcg_at_k(recommended, relevant, k=10):
    dcg   = sum(1.0 / np.log2(i + 2) for i, p in enumerate(recommended[:k]) if p in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_ranking(get_recs_fn, relevant_items, k=10, max_users=300):
    """
    Computes Precision@K, Recall@K and NDCG@K averaged over up to max_users.
    get_recs_fn(customer_id) must return [(product_id, score), ...].
    """
    precisions, recalls, ndcgs = [], [], []
    for customer_id in list(relevant_items.keys())[:max_users]:
        try:
            recs              = get_recs_fn(customer_id)
            recommended_ids   = [p for p, _ in recs]
            p, r              = precision_recall_at_k(recommended_ids, relevant_items[customer_id], k)
            n                 = ndcg_at_k(recommended_ids, relevant_items[customer_id], k)
            precisions.append(p)
            recalls.append(r)
            ndcgs.append(n)
        except Exception:
            continue

    def _mean(lst):
        return round(float(np.mean(lst)), 4) if lst else 0.0

    return {
        f"Precision@{k}": _mean(precisions),
        f"Recall@{k}":    _mean(recalls),
        f"NDCG@{k}":     _mean(ndcgs),
    }


# ─────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────

def plot_metrics(results: pd.DataFrame, k: int = 10, save_path: str = None):
    """
    Genera un gráfico de barras agrupadas comparando los modelos.
    Llama a esta función al final de cada ejecución para ver la evolución.

    results   : DataFrame devuelto por main() con columnas Model, Precision@K, Recall@K, NDCG@K
    k         : valor de K usado (solo para el título)
    save_path : si se indica, guarda el PNG en esa ruta además de mostrarlo
    """
    metrics   = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"]
    models    = results["Model"].tolist()
    n_models  = len(models)
    n_metrics = len(metrics)

    # Colores por métrica
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    x      = np.arange(n_models)
    width  = 0.22
    offset = width * (n_metrics - 1) / 2

    _, ax = plt.subplots(figsize=(max(8, n_models * 1.6), 5))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = results[metric].tolist()
        bars   = ax.bar(x + i * width - offset, values, width, label=metric, color=color, alpha=0.88)

        # Etiqueta de valor sobre cada barra
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0003,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Comparativa de modelos — Precision / Recall / NDCG @{k}", fontsize=13, pad=14)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.set_ylim(0, max(results[metrics].max()) * 1.35 + 1e-5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")

    plt.show()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main(k=50, max_iter=200, embedding_dim=50):
    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("=" * 55)
    print("Preparing data...")
    print("=" * 55)
    data           = prepare_all(train_ratio=0.8, min_product_orders=4, score_prior=5)
    relevant_items = get_test_relevant_items(data)
    print(f"\nTest users with train overlap: {len(relevant_items):,}")

    # ── 2. Train ──────────────────────────────────────────────────────────────
    models = m.train_all(data, k=embedding_dim, max_iter=max_iter)

    # ── 3. Recommendation wrappers ───────────────────────────────────────────
    dataset_cornac = data["dataset_cornac"]
    df_train       = data["df_train"]

    def cb_recommend(customer_id):
        products_seen = (
            df_train.loc[df_train["customer_unique_id"] == customer_id]
            .sort_values("order_purchase_timestamp")["product_id"]
            .tolist()
        )
        return models["content_based"].recommend(products_seen, top_k=k)

    def bpr_recommend(customer_id):
        return m.recommend_for_user_cornac(customer_id, models["bpr"], dataset_cornac, top_n=k)

    def mf_recommend(customer_id):
        return m.recommend_for_user_cornac(customer_id, models["mf"], dataset_cornac, top_n=k)

    def pop_recommend(customer_id):
        return m.recommend_for_user_cornac(customer_id, models["most_popular"], dataset_cornac, top_n=k)

    def random_recommend(customer_id):
        seen = df_train.loc[df_train["customer_unique_id"] == customer_id, "product_id"].tolist()
        return models["random"].recommend(customer_id, top_k=k, exclude_seen=seen)

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"Evaluating Precision@{k}, Recall@{k}, NDCG@{k}...")
    print("=" * 55)

    model_fns = [
        ("Content-Based", cb_recommend),
        ("BPR (Cornac)",  bpr_recommend),
        ("MF (Cornac)",   mf_recommend),
        ("Most Popular",  pop_recommend),
        ("Random",        random_recommend),
    ]

    rows = []
    for name, fn in model_fns:
        print(f"  · {name}...")
        metrics = evaluate_ranking(fn, relevant_items, k=k, max_users=300)
        rows.append({"Model": name, **metrics})

    # ── 5. Results table + chart ─────────────────────────────────────────────
    results = pd.DataFrame(rows)
    print("\n" + "=" * 55)
    print("COMPARATIVE RESULTS")
    print("=" * 55)
    print(results.to_string(index=False))
    print()

    plot_metrics(results, k=k, save_path="models/results_comparison.png")

    return results, models, data


if __name__ == "__main__":
    main(k=50)
