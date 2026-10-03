"""K-Means model selection and final training."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def evaluate_k_values(X, k_values=range(2, 9), random_state=42, n_init=20):
    rows = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
        )
        labels = model.fit_predict(X)

        rows.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette": silhouette_score(X, labels),
            "Calinski_Harabasz": calinski_harabasz_score(X, labels),
            "Davies_Bouldin": davies_bouldin_score(X, labels),
        })

    return pd.DataFrame(rows)


def select_k_by_silhouette(metrics: pd.DataFrame) -> int:
    return int(metrics.loc[metrics["Silhouette"].idxmax(), "K"])


def train_kmeans(X, n_clusters: int, random_state=42, n_init=20):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(X)
    return model, labels
