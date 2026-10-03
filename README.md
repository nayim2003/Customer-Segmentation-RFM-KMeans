# Customer Segmentation using RFM & K-Means

An end-to-end unsupervised machine learning project for segmenting customers based on **Recency, Frequency, and Monetary (RFM)** behavior.

## Objective

Transform raw retail transaction data into actionable behavioral customer segments using:

- transaction data cleaning
- RFM feature engineering
- log transformation
- feature scaling
- K-Means clustering
- multi-metric model selection
- cluster profiling
- business-oriented segment interpretation
- persisted model artifacts
- Streamlit inference and analysis

## Project Architecture

```text
customer-segmentation-rfm-kmeans/
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── rfm.py
│   ├── model_selection.py
│   ├── train.py
│   └── predict.py
├── artifacts/
├── outputs/
├── tests/
├── app.py
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data Cleaning

The finalized notebook rules are used:

- remove missing CustomerID
- parse transaction dates
- remove cancellation invoices
- retain positive Quantity
- retain positive UnitPrice
- remove duplicate rows
- calculate Revenue = Quantity × UnitPrice

### 2. RFM

- Recency = days since latest customer purchase
- Frequency = distinct invoice count
- Monetary = total customer revenue

### 3. Modeling

The modeling representation uses:

1. `log1p` transformation
2. `StandardScaler`
3. K-Means

K is evaluated from 2 through 8 using:

- Inertia
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

The default automated selection criterion is the highest Silhouette Score, followed by business/cluster-size review.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Train from the project root:

```bash
python -m src.train --data data/OnlineRetail.csv
```

This generates:

```text
artifacts/
├── rfm_scaler.joblib
├── kmeans_model.joblib
└── metadata.json

outputs/
├── customer_segments.csv
├── segment_profile.csv
└── model_selection_metrics.csv
```

## Streamlit

After training:

```bash
streamlit run app.py
```

The application supports:

### Transaction Segmentation

Upload a transaction CSV and receive:

- customer-level RFM
- cluster assignments
- business segments
- segment distribution
- downloadable results

### Single Customer Prediction

Enter:

- Recency
- Frequency
- Monetary

and the saved model predicts the corresponding segment.

## Important Modeling Note

K-Means cluster IDs are arbitrary. A cluster number does not inherently mean Gold, Silver, Diamond, etc.

Business segment names are therefore assigned from the observed cluster RFM profiles rather than from hard-coded cluster IDs.

## Analytical Notebook

`notebooks/Project02_RFM_Complete.ipynb` contains the full exploratory and analytical workflow, including markdown decisions and visual diagnostics.

## Testing

Run:

```bash
pytest -q
```

## Limitations

This is an unsupervised behavioral segmentation system. Segment labels are interpretations of historical transaction behavior and do not guarantee future customer response.

Future improvements could include:

- automated segment stability analysis
- alternative clustering algorithms
- customer lifetime value
- cohort analysis
- campaign response modeling
- monitoring for distribution drift

----
# ***Md. Nayim Howlader***
## ***BSc (Honours), Statistics,***
## ***Dhaka College, Dhaka.***
