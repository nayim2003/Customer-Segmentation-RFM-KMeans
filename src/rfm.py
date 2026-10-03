"""RFM feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = ["Recency", "Frequency", "Monetary"]


def build_rfm(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    reference_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        transactions.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    return rfm, reference_date


def transform_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Apply the finalized notebook log1p transformation."""
    transformed = rfm[FEATURES].copy()
    for col in FEATURES:
        transformed[col] = np.log1p(transformed[col])
    return transformed
