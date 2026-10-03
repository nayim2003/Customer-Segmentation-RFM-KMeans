import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import clean_transactions
from src.rfm import build_rfm
from src.predict import predict_segment


ARTIFACT_DIR = Path("artifacts")
OUTPUT_DIR = Path("outputs")

st.set_page_config(
    page_title="Customer Segmentation | RFM + K-Means",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Segmentation")
st.caption("RFM Analysis + K-Means Clustering")

if not (ARTIFACT_DIR / "kmeans_model.joblib").exists():
    st.error(
        "Trained model artifacts are missing. Run "
        "`python -m src.train --data data/OnlineRetail.csv` first."
    )
    st.stop()

with open(ARTIFACT_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = joblib.load(ARTIFACT_DIR / "kmeans_model.joblib")

tab1, tab2 = st.tabs(["📁 Transaction Segmentation", "👤 Single Customer"])

with tab1:
    st.subheader("Upload Transaction Data")
    uploaded = st.file_uploader(
        "Upload Online Retail CSV",
        type=["csv"],
    )

    if uploaded:
        raw = pd.read_csv(uploaded, encoding="unicode_escape")

        with st.spinner("Cleaning transactions and calculating RFM..."):
            clean = clean_transactions(raw)
            rfm, reference_date = build_rfm(clean)

            scaler = joblib.load(ARTIFACT_DIR / "rfm_scaler.joblib")

            transformed = np.log1p(
                rfm[["Recency", "Frequency", "Monetary"]]
            )
            X = scaler.transform(transformed)

            rfm["Cluster"] = model.predict(X)
            rfm["Segment"] = rfm["Cluster"].map(
                {int(k): v for k, v in metadata["segment_mapping"].items()}
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Customers", f"{len(rfm):,}")
        c2.metric("Transactions", f"{len(clean):,}")
        c3.metric("Segments", rfm["Segment"].nunique())

        st.subheader("Segment Distribution")
        distribution = (
            rfm["Segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )
        distribution["Percentage"] = (
            distribution["Customers"] / len(rfm) * 100
        )

        st.dataframe(distribution, use_container_width=True)

        st.subheader("Customer Segments")
        st.dataframe(
            rfm.sort_values("Monetary", ascending=False),
            use_container_width=True,
        )

        csv = rfm.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Segmented Customers",
            csv,
            "customer_segments.csv",
            "text/csv",
        )

with tab2:
    st.subheader("Predict a Customer Segment")

    col1, col2, col3 = st.columns(3)

    recency = col1.number_input("Recency (days)", min_value=0.0, value=30.0)
    frequency = col2.number_input("Frequency (invoices)", min_value=1.0, value=5.0)
    monetary = col3.number_input("Monetary value", min_value=0.01, value=500.0)

    if st.button("Predict Segment", type="primary"):
        result = predict_segment(
            recency,
            frequency,
            monetary,
            artifact_dir=str(ARTIFACT_DIR),
        )

        st.success(f"Predicted Segment: **{result['segment']}**")
        st.metric("Cluster", result["cluster"])
