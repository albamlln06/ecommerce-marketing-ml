import numpy as np
import cornac
from cornac.models import BPR, MF, MostPop
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# CONTENT-BASED
# ─────────────────────────────────────────────

class ContentBasedRecommender:
    """
    Recomendador basado en similitud coseno sobre la feature matrix de productos.
    Funciona con una sola interacción → ideal para cold-start en Olist.
    """

    def __init__(self, product_matrix, product_ids):
        self.product_matrix = product_matrix
        self.product_ids    = product_ids
        self.id_to_idx      = {pid: i for i, pid in enumerate(product_ids)}

    def fit(self):
        # Pre-calcular matriz de similitud completa
        # shape: (n_productos x n_productos)
        print("Calculando matriz de similitud coseno...")
        self.sim_matrix = cosine_similarity(self.product_matrix)
        np.fill_diagonal(self.sim_matrix, -1)  # excluir auto-similitud
        print(f"Similitud calculada: {self.sim_matrix.shape}")
        return self

    def recommend(self, product_ids_seen, top_k=10, exclude_seen=True):
        """
        Dado un conjunto de productos vistos/comprados,
        devuelve los top_k productos más similares.

        product_ids_seen : lista de product_id comprados por el usuario
        top_k            : número de recomendaciones
        exclude_seen     : excluir productos ya comprados
        """
        seen_indices = [
            self.id_to_idx[pid]
            for pid in product_ids_seen
            if pid in self.id_to_idx
        ]

        if not seen_indices:
            # Cold-start total: devolver los más populares por score
            scores = self.product_matrix[:, -1]  # última col = smoothed_score
        else:
            # Media de similitud con todos los productos vistos
            scores = self.sim_matrix[seen_indices].mean(axis=0)

        if exclude_seen:
            scores[seen_indices] = -1

        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.product_ids[i], float(scores[i])) for i in top_indices]


def train_content_based(product_matrix, product_ids):
    print("=" * 55)
    print("Entrenando Content-Based Recommender...")
    print("=" * 55)
    model = ContentBasedRecommender(product_matrix, product_ids)
    model.fit()
    print("✓ Content-Based listo.\n")
    return model


# ─────────────────────────────────────────────
# CORNAC — BPR
# ─────────────────────────────────────────────

def train_bpr(dataset_cornac, k=50, max_iter=200, learning_rate=0.01, seed=42):
    """
    BPR — Bayesian Personalized Ranking.
    Optimiza directamente el ranking relativo entre ítems.
    Equivalente al modo WARP de LightFM.
    Mejor opción para feedback implícito (compras).

    k             : dimensión de los embeddings
    max_iter      : épocas de entrenamiento
    learning_rate : tasa de aprendizaje
    """
    print("=" * 55)
    print("Entrenando BPR (Bayesian Personalized Ranking)...")
    print("=" * 55)

    model = BPR(
        k=k,
        max_iter=max_iter,
        learning_rate=learning_rate,
        lambda_reg=0.001,
        seed=seed,
        verbose=True
    )
    model.fit(dataset_cornac)
    print("✓ BPR listo.\n")
    return model


# ─────────────────────────────────────────────
# CORNAC — MF
# ─────────────────────────────────────────────

def train_mf(dataset_cornac, k=50, max_iter=200, learning_rate=0.01, seed=42):
    """
    MF — Matrix Factorization con SGD.
    Factoriza la matriz de interacciones en embeddings de usuario e ítem.
    Buen baseline colaborativo clásico para comparar con BPR.

    k             : dimensión de los embeddings
    max_iter      : épocas de entrenamiento
    learning_rate : tasa de aprendizaje
    """
    print("=" * 55)
    print("Entrenando MF (Matrix Factorization)...")
    print("=" * 55)

    model = MF(
        k=k,
        max_iter=max_iter,
        learning_rate=learning_rate,
        lambda_reg=0.001,
        use_bias=True,
        seed=seed,
        verbose=True
    )
    model.fit(dataset_cornac)
    print("✓ MF listo.\n")
    return model


# ─────────────────────────────────────────────
# CORNAC — MOST POPULAR (baseline)
# ─────────────────────────────────────────────

def train_most_popular(dataset_cornac):
    """
    Most Popular — recomienda siempre los productos más comprados.
    Baseline fuerte en datasets con poca sparsidad.
    Si los modelos CF no superan esto, hay un problema de señal.
    """
    print("=" * 55)
    print("Entrenando Most Popular (baseline)...")
    print("=" * 55)

    model = MostPop()
    model.fit(dataset_cornac)
    print("✓ Most Popular listo.\n")
    return model


# ─────────────────────────────────────────────
# RANDOM (baseline mínimo)
# ─────────────────────────────────────────────

class RandomRecommender:
    """
    Recomienda productos aleatorios del catálogo.
    Cota inferior: cualquier modelo debe superar esto.
    """

    def __init__(self, product_ids, seed=42):
        self.product_ids = product_ids
        self.rng         = np.random.default_rng(seed)

    def fit(self, *args, **kwargs):
        return self

    def recommend(self, user_id, top_k=10, exclude_seen=None):
        exclude_seen = exclude_seen or []
        candidates   = [p for p in self.product_ids if p not in set(exclude_seen)]
        chosen       = self.rng.choice(candidates, size=min(top_k, len(candidates)), replace=False)
        return [(pid, 0.0) for pid in chosen]


def train_random(product_ids, seed=42):
    print("=" * 55)
    print("Entrenando Random (baseline mínimo)...")
    print("=" * 55)
    model = RandomRecommender(product_ids, seed=seed)
    print("✓ Random listo.\n")
    return model


# ─────────────────────────────────────────────
# PIPELINE COMPLETA DE ENTRENAMIENTO
# ─────────────────────────────────────────────

def train_all(data, k=50, max_iter=200, learning_rate=0.01, seed=42):
    """
    Entrena todos los modelos y baselines en un único paso.

    Parámetros
    ----------
    data          : diccionario devuelto por prepare_all()
    k             : dimensión de embeddings para BPR y MF
    max_iter      : épocas de entrenamiento para BPR y MF
    learning_rate : tasa de aprendizaje para BPR y MF
    seed          : semilla para reproducibilidad

    Retorna
    -------
    Diccionario con todos los modelos entrenados listos para evaluar.
    """
    product_matrix   = data["product_matrix"]
    product_ids      = data["product_ids"]
    dataset_cornac   = data["dataset_cornac"]

    models = {}

    # Content-Based
    models["content_based"] = train_content_based(product_matrix, product_ids)

    # Cornac CF
    models["bpr"] = train_bpr(
        dataset_cornac,
        k=k,
        max_iter=max_iter,
        learning_rate=learning_rate,
        seed=seed
    )
    models["mf"] = train_mf(
        dataset_cornac,
        k=k,
        max_iter=max_iter,
        learning_rate=learning_rate,
        seed=seed
    )

    # Baselines
    models["most_popular"] = train_most_popular(dataset_cornac)
    models["random"]       = train_random(product_ids, seed=seed)

    print("=" * 55)
    print("✓ Todos los modelos entrenados.")
    print("=" * 55)
    print("Modelos disponibles:", list(models.keys()))

    return models


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import data_preprocess

    # Cargar datos
    data = data_preprocess.prepare_all()

    # Entrenar todos los modelos
    models = train_all(data)

    # Ejemplo de recomendación con Content-Based
    sample_user = data["df_train"]["customer_unique_id"].iloc[0]
    sample_products = (
        data["df_train"]
        [data["df_train"]["customer_unique_id"] == sample_user]
        ["product_id"]
        .tolist()
    )

    print(f"\nUsuario de ejemplo: {sample_user}")
    print(f"Productos comprados: {sample_products}")

    recs = models["content_based"].recommend(sample_products, top_k=5)
    print("\nRecomendaciones Content-Based:")
    for pid, score in recs:
        print(f"  {pid}  (similitud: {score:.4f})")

    # Ejemplo de recomendación con BPR
    cornac_uid = data["dataset_cornac"].uid_map.get(sample_user)
    if cornac_uid is not None:
        bpr_recs = models["bpr"].score(cornac_uid)
        top5_idx = np.argsort(bpr_recs)[::-1][:5]
        iid_map_inv = {v: k for k, v in data["dataset_cornac"].iid_map.items()}
        print("\nRecomendaciones BPR:")
        for idx in top5_idx:
            print(f"  {iid_map_inv.get(idx, idx)}  (score: {bpr_recs[idx]:.4f})")
