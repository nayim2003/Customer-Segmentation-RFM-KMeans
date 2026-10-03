"""Inference utilities for trained RFM segmentation artifacts."""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd


def load_artifacts(artifact_dir="artifacts"):
    scaler = joblib.load(f"{artifact_dir}/rfm_scaler.joblib")
    model = joblib.load(f"{artifact_dir}/kmeans_model.joblib")
    with open(f"{artifact_dir}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return scaler, model, metadata


def predict_segment(recency, frequency, monetary, artifact_dir="artifacts"):
    scaler, model, metadata = load_artifacts(artifact_dir)

    values = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
    }])

    values = np.log1p(values)
    X = scaler.transform(values)

    cluster = int(model.predict(X)[0])
    segment = metadata["segment_mapping"].get(str(cluster), f"Cluster {cluster}")

    return {"cluster": cluster, "segment": segment}
