import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score

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

def get_test_relevant_items(data, warm_only=False):
    """
    Returns {customer_id: set(product_ids)} for test users that also appear in train.

    warm_only=True  → only users seen by the CF models (dataset_cornac)
    warm_only=False → all train users (for Content-Based, which handles cold-start)
    """
    test = data["df_test"]
    if warm_only:
        train_users = set(data["dataset_cornac"].uid_map.keys())
    else:
        train_users = set(data["df_train"]["customer_unique_id"])
    return (
        test[test["customer_unique_id"].isin(train_users)]
        .groupby("customer_unique_id")["product_id"]
        .apply(set)
        .to_dict()
    )


def get_test_relevant_categories(data, warm_only=False):
    """
    Returns {customer_id: set(categories)} for test users that also appear in train.
    Relaxes evaluation from exact product match to category match.
    """
    test = data["df_test"]
    if warm_only:
        train_users = set(data["dataset_cornac"].uid_map.keys())
    else:
        train_users = set(data["df_train"]["customer_unique_id"])
    return (
        test[test["customer_unique_id"].isin(train_users)]
        .dropna(subset=["category"])
        .groupby("customer_unique_id")["category"]
        .apply(set)
        .to_dict()
    )


def build_product_category_map(data):
    """Returns {product_id: category} covering both train and test."""
    df = pd.concat([data["df_train"], data["df_test"]], ignore_index=True)
    return (
        df.dropna(subset=["category"])
        .drop_duplicates("product_id")
        .set_index("product_id")["category"]
        .to_dict()
    )


def evaluate_ranking_by_category(get_recs_fn, relevant_categories, product_to_category,
                                  k=10, max_users=300):
    """
    Like evaluate_ranking but maps recommended product_ids to categories before
    computing hits. Each category counts at most once per user (deduped by rank).
    """
    precisions, recalls, ndcgs = [], [], []
    for customer_id in list(relevant_categories.keys())[:max_users]:
        try:
            recs = get_recs_fn(customer_id)
            # Map products → categories, keeping first occurrence of each category
            seen_cats = set()
            recommended_cats = []
            for pid, _ in recs:
                cat = product_to_category.get(pid)
                if cat and cat not in seen_cats:
                    recommended_cats.append(cat)
                    seen_cats.add(cat)

            relevant = relevant_categories[customer_id]
            p, r = precision_recall_at_k(recommended_cats, relevant, k)
            n     = ndcg_at_k(recommended_cats, relevant, k)
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
        f"NDCG@{k}":      _mean(ndcgs),
    }


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
# AUROC
# ─────────────────────────────────────────

def auroc_content_based(cb_model, relevant_items, df_train, max_users=300):
    """
    AUROC a nivel de producto para Content-Based.
    Para cada usuario: puntúa todos los productos del catálogo y calcula
    AUROC entre los comprados en test (positivos) y el resto (negativos).
    """
    scores = []
    for user_id, pos_items in list(relevant_items.items())[:max_users]:
        products_seen = (
            df_train.loc[df_train["customer_unique_id"] == user_id]
            ["product_id"].tolist()
        )
        all_scores = cb_model.score_all(products_seen)           # (n_products,)
        all_ids    = cb_model.product_ids

        labels = [1 if pid in pos_items else 0 for pid in all_ids]
        if sum(labels) == 0 or sum(labels) == len(labels):
            continue  # necesita las dos clases para calcular AUROC

        scores.append(roc_auc_score(labels, all_scores))

    return round(float(np.mean(scores)), 4) if scores else 0.0


def auroc_cornac_by_category(model, dataset_cornac, relevant_cats,
                              product_to_category, max_users=300):
    """
    AUROC a nivel de categoría para modelos Cornac (BPR, MF, MostPop).
    Para cada warm user: obtiene los scores de todos los items del catálogo,
    los agrega por categoría (max) y calcula AUROC frente a las categorías
    compradas en test.
    """
    scores = []
    for user_id, pos_cats in list(relevant_cats.items())[:max_users]:
        if user_id not in dataset_cornac.uid_map:
            continue

        user_idx   = dataset_cornac.uid_map[user_id]
        item_scores = model.score(user_idx)  # array (n_items,)

        # Agrupar scores por categoría (máximo de los productos de esa categoría)
        cat_scores = {}
        for item_id, item_idx in dataset_cornac.iid_map.items():
            cat = product_to_category.get(item_id)
            if cat is None:
                continue
            s = float(item_scores[item_idx])
            if cat not in cat_scores or s > cat_scores[cat]:
                cat_scores[cat] = s

        if not cat_scores:
            continue

        cats   = list(cat_scores.keys())
        preds  = [cat_scores[c] for c in cats]
        labels = [1 if c in pos_cats else 0 for c in cats]

        if sum(labels) == 0 or sum(labels) == len(labels):
            continue

        scores.append(roc_auc_score(labels, preds))

    return round(float(np.mean(scores)), 4) if scores else 0.0


def plot_auroc(auroc_results: dict, save_path: str = None):
    """Gráfico de barras con el AUROC por modelo."""
    models = list(auroc_results.keys())
    values = list(auroc_results.values())
    colors = ["#4C72B0" if v > 0.5 else "#DD8452" for v in values]

    _, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 4))
    bars  = ax.bar(models, values, color=colors, alpha=0.88)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random baseline (0.5)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC por modelo", fontsize=13, pad=14)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en: {save_path}")

    plt.show()


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
    data = prepare_all(train_ratio=0.8, score_prior=5)

    relevant_items_all  = get_test_relevant_items(data, warm_only=False)
    relevant_items_warm = get_test_relevant_items(data, warm_only=True)
    relevant_cats_all   = get_test_relevant_categories(data, warm_only=False)
    relevant_cats_warm  = get_test_relevant_categories(data, warm_only=True)
    product_to_category = build_product_category_map(data)

    print(f"\nTest users (all train overlap): {len(relevant_items_all):,}")
    print(f"Test users (warm CF overlap):   {len(relevant_items_warm):,}")

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

    model_fns = [
        ("Content-Based", cb_recommend,     relevant_items_all,  relevant_cats_all),
        ("BPR (Cornac)",  bpr_recommend,    relevant_items_warm, relevant_cats_warm),
        ("MF (Cornac)",   mf_recommend,     relevant_items_warm, relevant_cats_warm),
        ("Most Popular",  pop_recommend,    relevant_items_warm, relevant_cats_warm),
        ("Random",        random_recommend, relevant_items_all,  relevant_cats_all),
    ]

    # ── 4a. Evaluate by product ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"Evaluating by PRODUCT  —  @{k}...")
    print("=" * 55)
    rows_product = []
    for name, fn, rel_items, _ in model_fns:
        print(f"  · {name}...")
        metrics = evaluate_ranking(fn, rel_items, k=k, max_users=300)
        rows_product.append({"Model": name, **metrics})

    # ── 4b. Evaluate by category ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"Evaluating by CATEGORY —  @{k}...")
    print("=" * 55)
    rows_cat = []
    for name, fn, _, rel_cats in model_fns:
        print(f"  · {name}...")
        metrics = evaluate_ranking_by_category(fn, rel_cats, product_to_category, k=k, max_users=300)
        rows_cat.append({"Model": name, **metrics})

    # ── 5. Results tables ─────────────────────────────────────────────────────
    results_product = pd.DataFrame(rows_product)
    results_cat     = pd.DataFrame(rows_cat)

    print("\n" + "=" * 55)
    print("RESULTS BY PRODUCT (exact match)")
    print("=" * 55)
    print(results_product.to_string(index=False))

    print("\n" + "=" * 55)
    print("RESULTS BY CATEGORY (relaxed match)")
    print("=" * 55)
    print(results_cat.to_string(index=False))

    # ── 6. AUROC ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Computing AUROC...")
    print("=" * 55)
    auroc_results = {
        "Content-Based\n(product)": auroc_content_based(
            models["content_based"], relevant_items_all, df_train, max_users=300
        ),
        "BPR\n(category)": auroc_cornac_by_category(
            models["bpr"], dataset_cornac, relevant_cats_warm, product_to_category
        ),
        "MF\n(category)": auroc_cornac_by_category(
            models["mf"], dataset_cornac, relevant_cats_warm, product_to_category
        ),
        "MostPop\n(category)": auroc_cornac_by_category(
            models["most_popular"], dataset_cornac, relevant_cats_warm, product_to_category
        ),
    }
    print("\nAUROC results:")
    for name, val in auroc_results.items():
        print(f"  {name.replace(chr(10), ' '):30s} {val:.4f}")

    # ── 7. Charts ─────────────────────────────────────────────────────────────
    plot_metrics(results_product, k=k, save_path="models/results_product.png")
    plot_metrics(results_cat,     k=k, save_path="models/results_category.png")
    plot_auroc(auroc_results,         save_path="models/results_auroc.png")

    return results_product, results_cat, auroc_results, models, data


if __name__ == "__main__":
    results_product, results_cat, auroc_results, models, data = main(k=50)
