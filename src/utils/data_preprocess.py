import sys
import os

sys.path.append(os.path.abspath(".."))

import pandas as pd
from src.utils.file_utils import fetch_file 


CUSTOMER_URL = "https://drive.google.com/file/d/1Js2P2OX65L0mfWRv63g7z-XAv30fHl7r/view?usp=share_link"
GEOLOCATION_URL = "https://drive.google.com/file/d/14pNzmp4Efr0NBJyCATceyDkRW_XzvTLQ/view?usp=share_link"
ORDER_ITEMS_URL = "https://drive.google.com/file/d/1jLHg1ePMZw9tZ1S_b8JZBjTlSWzpTRU-/view?usp=share_link"
ORDER_PAYMENTS_URL = "https://drive.google.com/file/d/1WAhjEt29Q3tsXqShpgMTq-6qNdeXM3St/view?usp=share_link"
REVIEWS_URL = "https://drive.google.com/file/d/15vPOEMjN2nDKs5_1I1fSaktsDE-e_ftW/view?usp=share_link"
ORDERS_URL = "https://drive.google.com/file/d/1jyFUxKFJFGZiHHf-7QuphbKvr5LsGtZR/view?usp=share_link"
PRODUCTS_URL = "https://drive.google.com/file/d/18GWfqup9MmsxA8KtD_-s-RYSyL1Ec-nm/view?usp=share_link"
SELLER_URL = "https://drive.google.com/file/d/1QNWrPqOn3Cxbj0qyOzm6p4Dsqp5SDyyg/view?usp=share_link"
CATEGORIES_URL = "https://drive.google.com/file/d/1fQuNKaCxOjMqsCOrhwpGQ7Y9slmrrUCC/view?usp=share_link"

def load_datasets():
    df_customer = fetch_file(CUSTOMER_URL)
    df_seller = fetch_file(SELLER_URL)
    df_geolocation = fetch_file(GEOLOCATION_URL)
    df_order_items = fetch_file(ORDER_ITEMS_URL)
    df_order_payments = fetch_file(ORDER_PAYMENTS_URL)
    df_reviews = fetch_file(REVIEWS_URL)
    df_orders = fetch_file(ORDERS_URL)
    df_products = fetch_file(PRODUCTS_URL)
    df_categories = fetch_file(CATEGORIES_URL)

    df_orders = order_process(df_orders)
    df_reviews = review_process(df_reviews)
    df_products = products_process(df_products, df_categories)
    df_products = clean_products(df_products, df_order_items, df_categories)

    return {
        "customer": df_customer,
        "seller": df_seller,
        "geolocation": df_geolocation,
        "order_items": df_order_items,
        "order_payments": df_order_payments,
        "reviews": df_reviews,
        "orders": df_orders,
        "products": df_products,
        "categories": df_categories
    }

def load_complete_dataset():
    datasets = load_datasets()

    df_complete = datasets["orders"].merge(datasets["customer"], on='customer_id', how='left')
    df_complete = df_complete.merge(datasets["order_items"], left_on='order_id', right_on='order_id', how='left')
    df_complete = df_complete.merge(datasets["products"], left_on='product_id', right_on='product_id', how='left')
    df_complete = df_complete.merge(datasets["categories"], left_on='product_category_name', right_on='product_category_name', how='left')
    df_complete = df_complete.merge(datasets["order_payments"], left_on='order_id', right_on='order_id', how='left')
    df_complete = df_complete.merge(datasets["reviews"], left_on='order_id', right_on='order_id', how='left')
    df_complete = df_complete.merge(datasets["seller"], left_on='seller_id', right_on='seller_id', how='left')

    return df_complete

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

def products_process(df_products, df_categories):
    df_products['product_category_name'] = (
        df_products['product_category_name'].fillna('unknown')
    )
    df_products = df_products.merge(df_categories, on='product_category_name', how='left')
    df_products['category'] = (
        df_products['product_category_name_english']
        .fillna(df_products['product_category_name'])
        .fillna('unknown')
    )
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

    #Rellenamos pedidos sin valorar con un score neutro (3) para que el modelo no los descarte por completo
    df['rating'] = df['review_score'].fillna(3.0)

    #Asignamos pesos más balanceados según el rating
    df['implicit_weight'] = df['rating'].map({
        1: 0.1, 2: 0.3, 3: 0.6, 4: 1.0, 5: 1.5
    }).fillna(0.6)

    return df

def clean_products(df_products: pd.DataFrame,
                   df_items: pd.DataFrame,
                   df_category: pd.DataFrame,
                   min_orders: int = 4) -> pd.DataFrame:


    product_freq = (
        df_items
        .groupby('product_id')['order_id']
        .nunique()
        .reset_index(name='n_orders')
    )

    valid_products = product_freq[product_freq['n_orders'] >= min_orders]['product_id']

    df_clean = df_products[df_products['product_id'].isin(valid_products)].copy()

    df_clean = df_clean.merge(df_category, on='product_category_name', how='left')
    df_clean['category'] = (
        df_clean['product_category_name_english']
        .fillna(df_clean['product_category_name'])
        .fillna('unknown')
    )
    df_clean = df_clean.drop(
        columns=['product_category_name', 'product_category_name_english'],
        errors='ignore'
    )

    numeric_cols = df_clean.select_dtypes(include='number').columns
    for col in numeric_cols:
        n_nulls = df_clean[col].isnull().sum()
        if n_nulls > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    df_clean = df_clean.merge(product_freq, on='product_id', how='left')

    total     = len(df_products)
    kept      = len(df_clean)
    removed   = total - kept

    print(total, "productos originales" )
    print(kept, "productos resultantes" )
    print(removed, "productos eliminados" )

    return df_clean

