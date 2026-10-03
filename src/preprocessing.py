"""Transaction cleaning utilities for the RFM segmentation project."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Online Retail transactions using the notebook's finalized rules."""
    validate_columns(df)

    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["InvoiceNo"] = data["InvoiceNo"].astype(str).str.strip()
    data["IsCancellation"] = data["InvoiceNo"].str.upper().str.startswith("C")

    data = data.dropna(subset=["CustomerID", "InvoiceDate"])
    data = data.loc[~data["IsCancellation"]]
    data = data.loc[data["Quantity"] > 0]
    data = data.loc[data["UnitPrice"] > 0]
    data = data.drop_duplicates()

    data["CustomerID"] = data["CustomerID"].astype(int)
    data["Revenue"] = data["Quantity"] * data["UnitPrice"]

    return data
