"""End-to-end training entry point.

Usage:
    python -m src.train --data data/OnlineRetail.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .preprocessing import clean_transactions
from .rfm import FEATURES, build_rfm, transform_rfm
from .model_selection import evaluate_k_values, select_k_by_silhouette, train_kmeans


RANDOM_STATE = 42
N_INIT = 20


def train(data_path, artifact_dir="artifacts", output_dir="outputs"):
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path, encoding="unicode_escape")
    clean = clean_transactions(raw)

    rfm, reference_date = build_rfm(clean)
    transformed = transform_rfm(rfm)

    scaler = StandardScaler()
    X = scaler.fit_transform(transformed)

    metrics = evaluate_k_values(
        X, k_values=range(2, 9),
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    final_k = select_k_by_silhouette(metrics)
    model, labels = train_kmeans(
        X, final_k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    rfm["Cluster"] = labels

    profile = rfm.groupby("Cluster")[FEATURES].mean()
    profile["RecencyRank"] = profile["Recency"].rank(ascending=True)
    profile["FrequencyRank"] = profile["Frequency"].rank(ascending=False)
    profile["MonetaryRank"] = profile["Monetary"].rank(ascending=False)
    profile["BusinessScore"] = (
        -profile["RecencyRank"]
        + profile["FrequencyRank"]
        + profile["MonetaryRank"]
    )
    profile = profile.sort_values("BusinessScore", ascending=False)

    ordered = profile.index.tolist()
    if len(ordered) == 3:
        names = {
            ordered[0]: "Champions",
            ordered[1]: "Loyal Customers",
            ordered[2]: "At-Risk / Low-Value Customers",
        }
    else:
        generic = [
            "Highest-Value Segment", "High-Value Segment",
            "Mid-Value Segment", "Lower-Value Segment",
            "At-Risk Segment", "Low-Value Segment",
            "Niche Segment", "Emerging Segment",
        ]
        names = {c: generic[i] if i < len(generic) else f"Segment {i+1}"
                 for i, c in enumerate(ordered)}

    rfm["Segment"] = rfm["Cluster"].map(names)

    segment_profile = (
        rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Median_Recency=("Recency", "median"),
            Avg_Frequency=("Frequency", "mean"),
            Median_Frequency=("Frequency", "median"),
            Avg_Monetary=("Monetary", "mean"),
            Median_Monetary=("Monetary", "median"),
        )
    )
    segment_profile["Customer_Percentage"] = (
        segment_profile["Customers"] / len(rfm) * 100
    )

    rfm.to_csv(output_dir / "customer_segments.csv", index=False)
    segment_profile.to_csv(output_dir / "segment_profile.csv")
    metrics.to_csv(output_dir / "model_selection_metrics.csv", index=False)

    joblib.dump(scaler, artifact_dir / "rfm_scaler.joblib")
    joblib.dump(model, artifact_dir / "kmeans_model.joblib")

    metadata = {
        "features": FEATURES,
        "transformation": "log1p",
        "scaling": "StandardScaler",
        "algorithm": "KMeans",
        "n_clusters": int(final_k),
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "reference_date": str(reference_date),
        "segment_mapping": {str(k): v for k, v in names.items()},
    }

    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "customers": len(rfm),
        "final_k": final_k,
        "metrics": metrics,
        "segment_profile": segment_profile,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    result = train(args.data, args.artifact_dir, args.output_dir)

    print(f"Customers: {result['customers']:,}")
    print(f"Final K: {result['final_k']}")
    print("Training completed successfully.")
