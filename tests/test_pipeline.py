import pandas as pd

from src.preprocessing import clean_transactions
from src.rfm import build_rfm, transform_rfm


def sample_data():
    return pd.DataFrame({
        "InvoiceNo": ["1", "1", "2", "C3", "4"],
        "StockCode": ["A", "B", "A", "A", "A"],
        "Description": ["x"] * 5,
        "Quantity": [2, 1, 3, 1, -1],
        "InvoiceDate": [
            "2011-01-01", "2011-01-01", "2011-01-03",
            "2011-01-04", "2011-01-05"
        ],
        "UnitPrice": [10, 20, 10, 10, 10],
        "CustomerID": [1, 1, 1, 2, 2],
        "Country": ["UK"] * 5,
    })


def test_cleaning_removes_invalid_transactions():
    cleaned = clean_transactions(sample_data())
    assert len(cleaned) == 3
    assert (cleaned["Quantity"] > 0).all()


def test_rfm_creation():
    cleaned = clean_transactions(sample_data())
    rfm, reference_date = build_rfm(cleaned)

    assert len(rfm) == 1
    assert set(["Recency", "Frequency", "Monetary"]).issubset(rfm.columns)

    transformed = transform_rfm(rfm)
    assert transformed.shape == (1, 3)
