# Anomaly-Detection-project

## Building Energy Anomaly Detection – How to Run

This project trains an anomaly detection pipeline on building energy data and exports predictions, charts, and summaries.

### 1. Prerequisites

- **Python**: 3.9 or higher
- **OS**: Tested on Windows 10

### 2. Create and activate a virtual environment (recommended)

In a terminal from the `major` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
```

If `python` points to Python 2 on your machine, use `python3` instead.

### 3. Install dependencies

Still in the `major` folder:

```bash
pip install -r requirement.txt
```

> Note: The file is named `requirement.txt` (singular) in this project.

### 4. Prepare the input data

By default, the script expects a preprocessed CSV named `feature_engineered_dataset.csv` in the `major` folder.

- **Option A (default)**: Keep your CSV as `feature_engineered_dataset.csv` and place it directly in the `major` folder.  
- **Option B (custom path)**: If your data file or directory has a different path or name, open `model.py` and update the `DATA_PATH` variable:

```python
DATA_PATH = "path/to/your_file_or_folder.csv"
```

The loader supports:
- A **single CSV file** path, or
- A **directory containing multiple CSVs** (it will concatenate them).

### 5. Run the pipeline

From the `major` folder, with your virtual environment activated:

```bash
python model.py
```

The script will:
- Load and preprocess the data
- Engineer time‑series features
- Train three anomaly detection models (Isolation Forest, LOF, Robust Covariance)
- Build an ensemble anomaly score
- Generate plots and business insights

### 6. Outputs

After a successful run, check:

- **`results/anomaly_predictions.csv`**: Main anomaly predictions with timestamps (and building IDs if present)
- **`results/model_agreement.png`**: Model agreement heatmap
- **`results/feature_importance.png`**: Top isolation forest feature importances
- **`results/timeseries_anomalies.png`**: Sample electricity time series with flagged anomalies
- **`results/vote_distribution.png`**: Distribution of anomaly votes across models
- **`results/anomaly_by_hour.png`**: Anomaly rate by hour of day
- **`results/anomaly_by_month.png`**: Anomaly rate by month
- **`results/building_summary.csv`** (if `building_id` exists): Per‑building anomaly summary

Trained model artifacts are saved under the **`models`** folder:

- `models/scaler.pkl`
- `models/isolation_forest.pkl`
- `models/robust_covariance.pkl`

### 7. Re-running with new data

To run the analysis on a new dataset:

1. Place the new CSV (or folder of CSVs) where you want.
2. Update `DATA_PATH` in `model.py` to point to the new location.
3. Re-run:

```bash
python model.py
```

New results will be written into the `results` folder, overwriting files with the same names.
