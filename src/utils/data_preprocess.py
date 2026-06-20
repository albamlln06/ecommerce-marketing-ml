import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
import cornac
from cornac.data import Dataset as CornacDataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "data_files")

CUSTOMER_PATH       = os.path.join(_DATA_DIR, "olist_customers_dataset.csv")
GEOLOCATION_PATH    = os.path.join(_DATA_DIR, "olist_geolocation_dataset.csv")
ORDER_ITEMS_PATH    = os.path.join(_DATA_DIR, "olist_order_items_dataset.csv")
ORDER_PAYMENTS_PATH = os.path.join(_DATA_DIR, "olist_order_payments_dataset.csv")
REVIEWS_PATH        = os.path.join(_DATA_DIR, "reviews_translated.csv")
ORDERS_PATH         = os.path.join(_DATA_DIR, "olist_orders_dataset.csv")
PRODUCTS_PATH       = os.path.join(_DATA_DIR, "olist_products_dataset.csv")
SELLER_PATH         = os.path.join(_DATA_DIR, "olist_sellers_dataset.csv")
CATEGORIES_PATH     = os.path.join(_DATA_DIR, "product_category_name_translation.csv")



def order_process(df_orders):
    df_orders = df_orders[df_orders['order_status'] == 'delivered'].copy()
    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        df_orders[col] = pd.to_datetime(df_orders[col])
    df_orders = df_orders.dropna(subset=['order_delivered_customer_date'])
    return df_orders


def review_process(df_reviews):
    df_reviews['review_answer_timestamp'] = pd.to_datetime(
        df_reviews['review_answer_timestamp']
    )
    df_reviews = (
        df_reviews
        .sort_values('review_answer_timestamp', ascending=False)
        .drop_duplicates(subset='order_id', keep='first')
    )
    return df_reviews


def translate_categories(df_products):
    """
    Reemplaza product_category_name por su nombre en inglés.
    Si no existe traducción, conserva el nombre original en portugués.
    """
    df_categories = pd.read_csv(CATEGORIES_PATH)
    df_categories.columns = df_categories.columns.str.strip().str.lstrip('﻿')

    translation = df_categories.set_index('product_category_name')['product_category_name_english']

    df_products = df_products.copy()
    df_products['product_category_name'] = (
        df_products['product_category_name']
        .fillna('unknown')
        .map(translation)
        .fillna(df_products['product_category_name'].fillna('unknown'))
    )
    return df_products


def products_process(df_products):
    df_products = translate_categories(df_products)
    df_products['category'] = df_products['product_category_name']
    return df_products


def iqr_filter(group, col, factor=3.0):
    Q1 = group[col].quantile(0.25)
    Q3 = group[col].quantile(0.75)
    IQR = Q3 - Q1
    return group[
        (group[col] >= Q1 - factor * IQR) &
        (group[col] <= Q3 + factor * IQR)
    ]


def clean_outliers(df, col, group_col='category', factor=3.0):
    return df.groupby(group_col, group_keys=False).apply(iqr_filter, col=col, factor=factor)


def clean_scoring(df):
    print(df['review_score'].value_counts(normalize=True))

    # Rellenamos pedidos sin valorar con score neutro (3)
    df['rating'] = df['review_score'].fillna(3.0)

    df['implicit_weight'] = df['rating'].map({
        1: 0.1, 2: 0.3, 3: 0.6, 4: 1.0, 5: 1.5
    }).fillna(0.6)

    return df


def clean_products(df_products: pd.DataFrame,
                   df_items: pd.DataFrame,
                   min_orders: int = 4) -> pd.DataFrame:

    product_freq = (
        df_items
        .groupby('product_id')['order_id']
        .nunique()
        .reset_index(name='n_orders')
    )

    valid_products = product_freq[product_freq['n_orders'] >= min_orders]['product_id']
    df_clean = df_products[df_products['product_id'].isin(valid_products)].copy()

    df_clean = df_clean.drop(
        columns=['product_category_name', 'product_category_name_english'],
        errors='ignore'
    )

    numeric_cols = df_clean.select_dtypes(include='number').columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    df_clean = df_clean.merge(product_freq, on='product_id', how='left')

    total   = len(df_products)
    kept    = len(df_clean)
    removed = total - kept
    print(f"{total} productos originales | {kept} resultantes | {removed} eliminados")

    return df_clean


def load_datasets(min_product_orders=1):
    df_customer       = pd.read_csv(CUSTOMER_PATH)
    df_seller         = pd.read_csv(SELLER_PATH)
    df_geolocation    = pd.read_csv(GEOLOCATION_PATH)
    df_order_items    = pd.read_csv(ORDER_ITEMS_PATH)
    df_order_payments = pd.read_csv(ORDER_PAYMENTS_PATH)
    df_reviews        = pd.read_csv(REVIEWS_PATH)
    df_orders         = pd.read_csv(ORDERS_PATH)
    df_products       = pd.read_csv(PRODUCTS_PATH)

    df_orders   = order_process(df_orders)
    df_reviews  = review_process(df_reviews)
    df_products = products_process(df_products)
    df_products = clean_products(df_products, df_order_items, min_orders=min_product_orders)

    return {
        "customer":       df_customer,
        "seller":         df_seller,
        "geolocation":    df_geolocation,
        "order_items":    df_order_items,
        "order_payments": df_order_payments,
        "reviews":        df_reviews,
        "orders":         df_orders,
        "products":       df_products,
    }


def load_complete_dataset(min_product_orders=2):
    datasets = load_datasets(min_product_orders=min_product_orders)

    df = datasets["orders"].merge(datasets["customer"],       on='customer_id',  how='left')
    df = df.merge(datasets["order_items"],                    on='order_id',     how='left')
    df = df.merge(datasets["products"],                       on='product_id',   how='left')
    df = df.merge(datasets["order_payments"],                 on='order_id',     how='left')
    df = df.merge(datasets["reviews"],                        on='order_id',     how='left')
    df = df.merge(datasets["seller"],                         on='seller_id',    how='left')

    df = clean_scoring(df)

    print(df.head())

    return df

#División train y test para modelos

def temporal_split(df, train_ratio=0.8):
    """
    Split cronológico sobre order_purchase_timestamp.
    NUNCA usar random split en sistemas de recomendación — causaría data leakage.
    """
    cutoff = df["order_purchase_timestamp"].quantile(train_ratio)
    train  = df[df["order_purchase_timestamp"] <= cutoff].copy()
    test   = df[df["order_purchase_timestamp"] >  cutoff].copy()

    print(f"Cutoff: {cutoff.date()}")
    print(f"Train:  {len(train):,} filas ({train['customer_unique_id'].nunique():,} usuarios)")
    print(f"Test:   {len(test):,} filas  ({test['customer_unique_id'].nunique():,} usuarios)")

    return train, test

def get_product_avg_rating(df_products, df_order_items, df_order_reviews):
    
    items_reviews = df_order_items.merge(
        df_order_reviews[['order_id', 'review_score']],
        on='order_id',
        how='inner'
    )
    
    product_ratings = (
        items_reviews
        .groupby('product_id')['review_score']
        .agg(avg_review_score='mean', n_reviews='count')
        .reset_index()
    )
    
    df_products_with_rating = df_products.merge(
        product_ratings,
        on='product_id',
        how='left'
    )
    
    return df_products_with_rating

def clustering_preprocess():
    """
    Preprocesamiento para clustering de productos.
    Escalado de atributos numéricos y one-hot encoding de categoría.
    Si se pasa df_train, calcula avg_price y avg_review_score solo con datos
    de train para evitar leakage.
    """
    numeric_cols = [
        'avg_price',
        'product_weight_g',
        'product_length_cm',
        'product_height_cm',
        'product_width_cm',
        'avg_review_score'
    ]

    datasets = load_datasets()
    df_products = datasets["products"]

    df_order_items = datasets["order_items"]
    df_reviews = datasets["reviews"]
    df_products = get_product_avg_rating(df_products, df_order_items, df_reviews)
    df_products = df_products.drop_duplicates(subset='product_id').copy()

    avg_price = df_order_items.groupby('product_id')['price'].mean().rename('avg_price')
    df_products = df_products.merge(avg_price, on='product_id', how='left')

    df_products[numeric_cols] = df_products[numeric_cols].fillna(
        df_products[numeric_cols].median()
    )

    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(df_products[numeric_cols])
    print("Shape numéricas:", X_numeric.shape)

    X_category = pd.get_dummies(
        df_products['category'],
        prefix='cat'
    ).values
    print("Shape categoría:", X_category.shape)

    X_final = np.hstack([X_numeric, X_category])
    print("Shape final:", X_final.shape)

    product_ids = df_products['product_id'].values

    return X_final, product_ids, scaler, df_products



def find_optimal_k(X, k_range=range(2, 15)):

    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        
        print(f"K={k} | Inertia={kmeans.inertia_:.2f} | Silhouette={silhouettes[-1]:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(list(k_range), inertias, marker='o')
    axes[0].set_title('Elbow Method (Inertia)')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Inertia')
    
    axes[1].plot(list(k_range), silhouettes, marker='o', color='orange')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.show()
    
    return inertias, silhouettes

def fit_product_clustering(X, k, product_ids):
    """
    Entrena KMeans con el K elegido y devuelve un DataFrame
    con product_id y su cluster asignado.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    df_clusters = pd.DataFrame({
        'product_id': product_ids,
        'cluster': cluster_labels
    })
    
    return df_clusters, kmeans

def inspect_clusters(df_products, df_clusters, numeric_cols, category_col='category_name', top_n_categories=3):
    """
    Une los clusters con las features originales del producto y describe
    cada cluster: medias de las variables numéricas y categorías más comunes.
    
    Parameters
    ----------
    df_products : DataFrame con product_id + columnas numéricas + categoría
    df_clusters : DataFrame con columnas ['product_id', 'cluster']
    numeric_cols : list de columnas numéricas usadas en el clustering
    category_col : nombre de la columna de categoría de producto
    top_n_categories : cuántas categorías top mostrar por cluster
    
    Returns
    -------
    df_merged : DataFrame combinado con la columna 'cluster' añadida
    summary : DataFrame con la media de cada feature numérica por cluster
    """
    
    df_merged = df_products.merge(df_clusters, on='product_id', how='inner')
    
    cluster_sizes = df_merged['cluster'].value_counts().sort_index()
    print("=== Tamaño de cada cluster ===")
    print(cluster_sizes)
    print(f"\nTotal de productos: {len(df_merged)}\n")
    
    summary = df_merged.groupby('cluster')[numeric_cols].mean().round(2)
    print("=== Medias de features numéricas por cluster ===")
    print(summary)
    print()
    
    print("=== Categorías más frecuentes por cluster ===")
    for c in sorted(df_merged['cluster'].unique()):
        cluster_data = df_merged[df_merged['cluster'] == c]
        top_cats = cluster_data[category_col].value_counts(normalize=True).head(top_n_categories) * 100
        
        print(f"\nCluster {c} (n={len(cluster_data)}):")
        for cat, pct in top_cats.items():
            print(f"  {cat}: {pct:.1f}%")
    
    return df_merged, summary


# funciones testing --------------------------------

def build_product_scores(train, C=5):
    """
    Suavizado bayesiano de la calificación por producto.
    Productos con pocas reviews convergen hacia la media global.
    Se calcula SOLO sobre train para evitar leakage.

    C: peso del prior (más alto → más suavizado).
    """
    global_mean = train["rating"].mean()

    product_scores = (
        train.groupby("product_id")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_score", "count": "n_reviews"})
    )
    product_scores["smoothed_score"] = (
        (product_scores["n_reviews"] * product_scores["avg_score"] + C * global_mean)
        / (product_scores["n_reviews"] + C)
    )

    print(f"Scores calculados para {len(product_scores):,} productos | Media global: {global_mean:.3f}")
    return product_scores[["smoothed_score"]], global_mean


def build_geo_encodings(train):
    """
    Target encoding geográfico: media de rating por seller_city y customer_state.
    Los valores desconocidos en test recibirán la media global como fallback.
    Se calcula SOLO sobre train para evitar leakage.
    """
    seller_enc   = train.groupby("seller_city")["rating"].mean().rename("seller_city_score")
    customer_enc = train.groupby("customer_state")["rating"].mean().rename("customer_state_score")

    print(f"Geo encoding: {len(seller_enc)} ciudades de vendedor | {len(customer_enc)} estados de cliente")
    return seller_enc, customer_enc


def build_product_feature_matrix(df, product_scores, seller_enc, customer_enc,
                                  global_mean, ohe=None, scaler=None, fit=True):
    """
    Construye la feature matrix normalizada (L2) por producto.

    Parámetros
    ----------
    df             : DataFrame con una fila por interacción (train o test)
    product_scores : Serie con smoothed_score por product_id (calculada en train)
    seller_enc     : Target encoding de seller_city (calculado en train)
    customer_enc   : Target encoding de customer_state (calculado en train)
    global_mean    : Media global de rating (calculada en train)
    ohe            : OneHotEncoder pre-fitteado (None si fit=True)
    scaler         : StandardScaler pre-fitteado (None si fit=True)
    fit            : True → entrena encoders (usar con train)
                     False → solo transforma (usar con test/inference)

    Retorna
    -------
    matrix_norm  : np.ndarray (n_productos x n_features), normalizado L2
    product_ids  : lista de product_id en el mismo orden que las filas
    ohe          : OneHotEncoder fitteado
    scaler       : StandardScaler fitteado
    """
    # Una fila por producto único
    catalog = df.drop_duplicates("product_id").reset_index(drop=True)

    # ── 1. Categoría → One-Hot Encoding (peso x1) ──
    cat_col = catalog[["category"]].fillna("unknown")
    if fit:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe.fit(cat_col)
    cat_features = ohe.transform(cat_col)

    # ── 2. Geográfico → Target Encoding (peso x1) ──
    geo_features = (
        catalog
        .join(seller_enc,   on="seller_city",    how="left")
        .join(customer_enc, on="customer_state",  how="left")
        [["seller_city_score", "customer_state_score"]]
        .fillna(global_mean)
        .values
    )

    # ── 3. Atributos físicos → StandardScaler (peso x0.5) ──
    quant_cols = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]
    quant_data = catalog[quant_cols].fillna(catalog[quant_cols].median())
    if fit:
        scaler = StandardScaler()
        scaler.fit(quant_data)
    quant_features = scaler.transform(quant_data) * 0.5

    # ── 4. Calificación suavizada (peso x2.0) ──
    score_features = (
        catalog[["product_id"]]
        .merge(product_scores, on="product_id", how="left")
        ["smoothed_score"]
        .fillna(global_mean)
        .values
        .reshape(-1, 1)
    ) * 2.0

    # ── Concatenar y normalizar L2 ──
    matrix      = np.hstack([cat_features, geo_features, quant_features, score_features])
    matrix_norm = normalize(matrix, norm="l2")

    product_ids = catalog["product_id"].tolist()
    print(f"Feature matrix: {len(product_ids):,} productos x {matrix_norm.shape[1]} features")

    return matrix_norm, product_ids, ohe, scaler


# ─────────────────────────────────────────────
# K-CORE FILTERING
# ─────────────────────────────────────────────

def kcore_filter(df, min_user_interactions=2, min_product_orders=4, max_iter=10):
    """
    Filtra iterativamente hasta que todos los usuarios tienen >= min_user_interactions
    pedidos únicos y todos los productos tienen >= min_product_orders pedidos únicos.

    Aplicar solo sobre train para evitar leakage. Los usuarios excluidos
    (con una sola compra) se tratan como cold-start y reciben recomendaciones
    basadas en contenido o popularidad.
    """
    for i in range(max_iter):
        n_before = len(df)

        product_counts = df.groupby("product_id")["order_id"].nunique()
        valid_products = product_counts[product_counts >= min_product_orders].index
        df = df[df["product_id"].isin(valid_products)]

        user_counts = df.groupby("customer_unique_id")["order_id"].nunique()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["customer_unique_id"].isin(valid_users)]

        n_after = len(df)
        print(f"  k-core iter {i+1}: {n_before:,} → {n_after:,} filas "
              f"({df['customer_unique_id'].nunique():,} usuarios, "
              f"{df['product_id'].nunique():,} productos)")

        if n_after == n_before:
            break

    return df


# ─────────────────────────────────────────────
# MATRICES DE INTERACCIÓN PARA LIGHTFM
# ─────────────────────────────────────────────

def build_interaction_matrix(df):
    """
    Matriz usuario x producto en formato sparse (CSR).
    Usa implicit_weight de clean_scoring como valor de interacción
    en lugar de 1s binarios, para reflejar la calidad de la interacción.
    """
    user_ids = df["customer_unique_id"].unique().tolist()
    item_ids = df["product_id"].unique().tolist()

    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {p: i for i, p in enumerate(item_ids)}

    rows = df["customer_unique_id"].map(user_index)
    cols = df["product_id"].map(item_index)
    data = df["implicit_weight"].values

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(item_ids))
    )

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print(f"Interaction matrix: {matrix.shape} | Densidad: {density:.6f}")

    return matrix, user_ids, item_ids, user_index, item_index



def build_cornac_dataset(train):
    """
    Prepara los objetos nativos de Cornac.
    Cornac espera una lista de tuplas (user_id, item_id, rating).
    Usamos implicit_weight como valor de interacción.
    """
    # Lista de tuplas (usuario, producto, peso)
    interactions = list(
        train[["customer_unique_id", "product_id", "implicit_weight"]]
        .itertuples(index=False, name=None)
    )

    # Dataset de Cornac
    dataset = CornacDataset.from_uir(interactions, seed=42)

    print(f"Cornac dataset: {dataset.num_users} usuarios | "
          f"{dataset.num_items} items | "
          f"{dataset.num_ratings} interacciones")

    return dataset, interactions

# ─────────────────────────────────────────────
# PIPELINE COMPLETA
# ─────────────────────────────────────────────

def prepare_all(train_ratio=0.8, min_product_orders=2, min_user_interactions=2, score_prior=5):
    """
    Ejecuta la pipeline completa de preparación de datos.

    Retorna un diccionario con todos los objetos listos para entrenar
    Content-Based y modelos CF (Cornac BPR/MF/MostPop).

    Parámetros
    ----------
    train_ratio            : fracción temporal para train (default 0.8)
    min_product_orders     : mínimo de pedidos únicos para incluir un producto (default 4)
    min_user_interactions  : mínimo de pedidos únicos para incluir un usuario en CF (default 2)
    score_prior            : peso del prior en suavizado bayesiano (default 5)

    Usuarios warm  → ≥ min_user_interactions compras → entran en los modelos CF
    Usuarios cold  → 1 sola compra → fallback a Content-Based o MostPop
    """
    print("=" * 55)
    print("1. Cargando datasets...")
    print("=" * 55)
    df = load_complete_dataset(min_product_orders=min_product_orders)
    print(df.head())

    print("\n" + "=" * 55)
    print("2. Split temporal...")
    print("=" * 55)
    train, test = temporal_split(df, train_ratio=train_ratio)

    print("\n" + "=" * 55)
    print("3. K-core filtering (train)...")
    print("=" * 55)
    train_cf = kcore_filter(
        train,
        min_user_interactions=min_user_interactions,
        min_product_orders=min_product_orders,
    )
    cold_users = set(train["customer_unique_id"].unique()) - set(train_cf["customer_unique_id"].unique())
    print(f"  Usuarios warm (CF):  {train_cf['customer_unique_id'].nunique():,}")
    print(f"  Usuarios cold-start: {len(cold_users):,}  → fallback Content-Based / MostPop")

    print("\n" + "=" * 55)
    print("4. Calificación suavizada por producto...")
    print("=" * 55)
    # Se calcula sobre train_cf para que las estadísticas reflejen solo usuarios con historial
    product_scores, global_mean = build_product_scores(train_cf, C=score_prior)

    print("\n" + "=" * 55)
    print("5. Encoding geográfico...")
    print("=" * 55)
    seller_enc, customer_enc = build_geo_encodings(train_cf)

    print("\n" + "=" * 55)
    print("6. Feature matrix de productos (train completo)...")
    print("=" * 55)
    # Se construye sobre train completo para cubrir también los productos de cold-start
    product_matrix, product_ids, ohe, scaler = build_product_feature_matrix(
        train, product_scores, seller_enc, customer_enc,
        global_mean, fit=True
    )

    print("\n" + "=" * 55)
    print("7. Matriz de interacciones sparse (CF)...")
    print("=" * 55)
    interaction_matrix, user_ids, item_ids, user_index, item_index = build_interaction_matrix(train_cf)

    print("\n" + "=" * 55)
    print("8. Dataset nativo Cornac (CF puro + híbrido)...")
    print("=" * 55)
    dataset_cornac, interactions_cornac = build_cornac_dataset(train_cf)
    print("\n✓ Pipeline completada.\n")

    return {
        # DataFrames
        "df_train"          : train,       # train completo (incluye cold-start)
        "df_train_cf"       : train_cf,    # solo usuarios warm, para modelos CF
        "df_test"           : test,
        "cold_users"        : cold_users,  # set de customer_unique_id sin historial suficiente

        # Encoders (fitteados en train_cf, aplicar a test con fit=False)
        "ohe"               : ohe,
        "scaler"            : scaler,
        "product_scores"    : product_scores,
        "seller_enc"        : seller_enc,
        "customer_enc"      : customer_enc,
        "global_mean"       : global_mean,

        # Content-Based (cubre catálogo completo, útil para cold-start)
        "product_matrix"    : product_matrix,
        "product_ids"       : product_ids,

        # CF sparse (solo usuarios warm)
        "interaction_matrix": interaction_matrix,
        "user_ids"          : user_ids,
        "item_ids"          : item_ids,
        "user_index"        : user_index,
        "item_index"        : item_index,

        # Cornac (solo usuarios warm)
        "dataset_cornac"     : dataset_cornac,
        "interactions_cornac": interactions_cornac,
    }


if __name__ == "__main__":
    data = prepare_all(
        train_ratio=0.8,
        min_product_orders=2,
        min_user_interactions=2,
        score_prior=5
    )

    print("Objetos disponibles:")
    for k, v in data.items():
        if hasattr(v, "shape"):
            print(f"  {k:25s} → shape {v.shape}")
        elif isinstance(v, list):
            print(f"  {k:25s} → lista de {len(v):,} elementos")
        elif isinstance(v, dict):
            print(f"  {k:25s} → dict de {len(v):,} elementos")
        elif isinstance(v, set):
            print(f"  {k:25s} → set de {len(v):,} elementos")
        else:
            print(f"  {k:25s} → {type(v).__name__}")
