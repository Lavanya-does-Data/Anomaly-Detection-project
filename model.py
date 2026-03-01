"""
=============================================================
Building Energy Anomaly Detection - Full ML Pipeline
=============================================================
Dataset  : Building Data Genome Project 2 (BDG2)
Author   : Evoastra Internship Project
Pipeline : EDA → Preprocessing → Feature Eng → Models → Evaluation → Business Insights
=============================================================
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import classification_report
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid")
os.makedirs("results", exist_ok=True)
os.makedirs("models",  exist_ok=True)

print("✅ Imports loaded successfully")


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """
    Accepts either:
      - A single pre-processed CSV (e.g. the sample you shared)
      - A directory containing multiple BDG2 meter CSVs
    Returns a single DataFrame sorted by timestamp.
    """
    if os.path.isdir(path):
        dfs = []
        for f in os.listdir(path):
            if f.endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(path, f)))
        df = pd.concat(dfs, ignore_index=True)
        print(f"📂 Loaded {len(dfs)} CSV files → {df.shape[0]:,} rows")
    else:
        df = pd.read_csv(path)
        print(f"📄 Loaded single CSV → {df.shape[0]:,} rows")

    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── CHANGE THIS PATH ──────────────────────────────────────────────────────────
# Option A – point to your extracted BDG2 directory:
#   DATA_PATH = "building-data-genome-project-2/data/meters/raw/"
# Option B – point to your single pre-processed CSV:
#   DATA_PATH = "your_file.csv"
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "feature_engineered_dataset.csv"   # <── UPDATE THIS

df_raw = load_data(DATA_PATH)
print(f"\n{'='*55}")
print(f"Shape       : {df_raw.shape}")
print(f"Columns     : {list(df_raw.columns)}")
print(f"Date range  : {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")
print(f"Buildings   : {df_raw['building_id'].nunique() if 'building_id' in df_raw.columns else 'N/A'}")
print(f"Missing %   :\n{(df_raw.isnull().mean()*100).round(2).to_string()}")


# ─────────────────────────────────────────────
# 2. BUILDING SELECTION
# ─────────────────────────────────────────────
# Strategy: pick the building(s) with the most complete electricity data.
# You can override this by setting SELECTED_BUILDINGS manually.

ENERGY_COLS = ["electricity", "chilled_water", "steam", "hot_water", "gas", "water"]
WEATHER_COLS = ["temperature", "humidity"] if "humidity" in df_raw.columns else ["temperature"]

if "building_id" in df_raw.columns:
    # Count non-null electricity records per building
    coverage = (
        df_raw.groupby("building_id")["electricity"]
        .apply(lambda x: x.notna().sum())
        .sort_values(ascending=False)
    )
    # Choose top-5 buildings for a representative but fast analysis
    SELECTED_BUILDINGS = coverage.head(5).index.tolist()
    print(f"\n🏢 Top-5 buildings by data coverage:\n{coverage.head(5).to_string()}")
    df = df_raw[df_raw["building_id"].isin(SELECTED_BUILDINGS)].copy()
else:
    df = df_raw.copy()

print(f"\n📊 Working dataset: {df.shape[0]:,} rows")


# ─────────────────────────────────────────────
# 3. CLEANING & PREPROCESSING
# ─────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Forward-fill then back-fill for time-series continuity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (
        df[numeric_cols]
        .ffill()
        .bfill()
        .fillna(0)
    )

    # Drop rows where electricity is still 0 AND all other energy cols are 0
    # (likely offline / sensor-off periods) — keep for now, just flag them
    df["all_zero"] = (df[ENERGY_COLS].fillna(0).sum(axis=1) == 0).astype(int)

    # Cap outliers at 1st–99th percentile per energy column
    for col in ENERGY_COLS:
        if col in df.columns:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    return df


df = clean(df)
print("✅ Cleaning complete")
print(f"   All-zero rows: {df['all_zero'].sum():,} ({df['all_zero'].mean()*100:.1f}%)")


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    W = 168  # 7-day rolling window (hours)

    for col in ENERGY_COLS:
        if col not in df.columns:
            continue

        # Rolling statistics
        df[f"{col}_rmean"] = df[col].rolling(W, min_periods=1).mean()
        df[f"{col}_rstd"]  = df[col].rolling(W, min_periods=1).std().fillna(0)

        # Z-score deviation from rolling baseline
        df[f"{col}_dev"] = (
            (df[col] - df[f"{col}_rmean"]) / (df[f"{col}_rstd"] + 1e-5)
        )

        # Lag features
        df[f"{col}_lag1"]  = df[col].shift(1)
        df[f"{col}_lag24"] = df[col].shift(24)
        df[f"{col}_lag168"]= df[col].shift(168)

    # Time features
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hour"] = df["hour"].between(8, 18).astype(int)

    # Interaction: temperature × electricity
    if "temperature" in df.columns:
        df["temp_x_elec"] = df["temperature"] * df["electricity"]

    df = df.fillna(0)
    return df


df = engineer_features(df)

# Cast time columns to int to prevent float index errors downstream
for _col in ["hour", "day_of_week", "month", "is_weekend", "is_business_hour"]:
    if _col in df.columns:
        df[_col] = df[_col].astype(int)

print(f"✅ Feature engineering complete — {df.shape[1]} total columns")


# ─────────────────────────────────────────────
# 5. NORMALIZE FEATURES FOR MODELS
# ─────────────────────────────────────────────
FEATURE_COLS = (
    [c for c in ENERGY_COLS if c in df.columns] +
    WEATHER_COLS +
    ["hour", "day_of_week", "month", "is_weekend", "is_business_hour"] +
    [c for c in df.columns if "_dev" in c or "_rmean" in c]
)
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

scaler = MinMaxScaler()
X = pd.DataFrame(
    scaler.fit_transform(df[FEATURE_COLS]),
    columns=FEATURE_COLS,
    index=df.index
)
joblib.dump(scaler, "models/scaler.pkl")
print(f"✅ Scaled {len(FEATURE_COLS)} features")


# ─────────────────────────────────────────────
# 6. ANOMALY DETECTION MODELS
# ─────────────────────────────────────────────
CONTAMINATION = 0.05   # Expected ~5% anomaly rate

print("\n🔍 Training anomaly detection models...")

# Model 1: Isolation Forest
iso = IsolationForest(
    contamination=CONTAMINATION,
    n_estimators=200,
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)
pred_iso = iso.fit_predict(X)           # +1 = normal, -1 = anomaly
score_iso = iso.decision_function(X)    # lower = more anomalous
joblib.dump(iso, "models/isolation_forest.pkl")
print("  ✓ Isolation Forest trained")

# Model 2: Local Outlier Factor
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=CONTAMINATION,
    n_jobs=-1
)
pred_lof = lof.fit_predict(X)
print("  ✓ Local Outlier Factor trained")

# Model 3: Robust Covariance (EllipticEnvelope)
# Subset to avoid memory issues on large datasets
SAMPLE_SIZE = min(50_000, len(X))
idx_sample = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
robust = EllipticEnvelope(contamination=CONTAMINATION, random_state=42)
robust.fit(X.iloc[idx_sample])
pred_robust = robust.predict(X)
joblib.dump(robust, "models/robust_covariance.pkl")
print("  ✓ Robust Covariance trained")

# ── Ensemble: majority vote (anomaly if ≥ 2 models agree) ──
df["vote_iso"]    = (pred_iso    == -1).astype(int)
df["vote_lof"]    = (pred_lof    == -1).astype(int)
df["vote_robust"] = (pred_robust == -1).astype(int)
df["anomaly_votes"] = df["vote_iso"] + df["vote_lof"] + df["vote_robust"]
df["is_anomaly"]    = (df["anomaly_votes"] >= 2).astype(int)
df["anomaly_score"] = -score_iso   # higher = more anomalous

n_anom = df["is_anomaly"].sum()
print(f"\n  📌 Ensemble detected {n_anom:,} anomalies "
      f"({n_anom/len(df)*100:.1f}% of records)")


# ─────────────────────────────────────────────
# 7. EVALUATION & VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n📊 Generating visualizations...")

# ── 7a: Model Agreement Heatmap ──────────────
agreement = pd.DataFrame({
    "IsolationForest": df["vote_iso"],
    "LOF":             df["vote_lof"],
    "RobustCov":       df["vote_robust"],
    "Ensemble":        df["is_anomaly"],
})
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    agreement.corr(), annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
    vmin=0, vmax=1, linewidths=0.5
)
ax.set_title("Model Agreement Correlation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/model_agreement.png", dpi=150)
plt.close()

# ── 7b: Feature Importance (Isolation Forest) ──
# IsolationForest has no .feature_importances_ — average across all trees instead
importances = np.mean([tree.feature_importances_ for tree in iso.estimators_], axis=0)
top_n = 15
top_idx = np.argsort(importances)[-top_n:]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(
    [FEATURE_COLS[i] for i in top_idx],
    importances[top_idx],
    color=sns.color_palette("Blues_d", top_n)
)
ax.set_xlabel("Importance Score")
ax.set_title("Top-15 Feature Importances — Isolation Forest", fontweight="bold")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
plt.close()

# ── 7c: Electricity Time-Series with Anomalies ──
fig, ax = plt.subplots(figsize=(14, 5))
sample = df.head(2000)
anomalies_sample = sample[sample["is_anomaly"] == 1]

ax.plot(sample["timestamp"], sample["electricity"],
        color="#2196F3", linewidth=0.8, label="Electricity (kWh)", alpha=0.85)
ax.scatter(anomalies_sample["timestamp"], anomalies_sample["electricity"],
           color="red", s=30, zorder=5, label="Anomaly", alpha=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=30)
ax.set_ylabel("Electricity (normalized kWh)")
ax.set_title("Electricity Consumption with Detected Anomalies", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("results/timeseries_anomalies.png", dpi=150)
plt.close()

# ── 7d: Anomaly Vote Distribution ──
fig, ax = plt.subplots(figsize=(6, 4))
vote_counts = df["anomaly_votes"].value_counts().sort_index()
bars = ax.bar(
    ["0 models\n(Normal)", "1 model", "2 models", "3 models\n(Strong)"],
    vote_counts.reindex([0, 1, 2, 3], fill_value=0),
    color=["#4CAF50", "#FFC107", "#FF5722", "#B71C1C"]
)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f"{bar.get_height():,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of Records")
ax.set_title("Anomaly Vote Distribution Across Models", fontweight="bold")
plt.tight_layout()
plt.savefig("results/vote_distribution.png", dpi=150)
plt.close()

# ── 7e: Anomaly by Hour of Day ──
fig, ax = plt.subplots(figsize=(10, 4))
hourly = df.groupby("hour")["is_anomaly"].mean() * 100
ax.bar(hourly.index, hourly.values, color=sns.color_palette("Reds_d", 24))
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Anomaly Rate (%)")
ax.set_title("Anomaly Rate by Hour of Day", fontweight="bold")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig("results/anomaly_by_hour.png", dpi=150)
plt.close()

# ── 7f: Anomaly by Month ──
fig, ax = plt.subplots(figsize=(10, 4))
monthly = df.groupby("month")["is_anomaly"].mean() * 100
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
ax.bar(monthly.index, monthly.values, color=sns.color_palette("Blues_d", 12))
ax.set_xticks(monthly.index)
ax.set_xticklabels([month_labels[int(m)-1] for m in monthly.index])
ax.set_ylabel("Anomaly Rate (%)")
ax.set_title("Anomaly Rate by Month (Seasonal Pattern)", fontweight="bold")
plt.tight_layout()
plt.savefig("results/anomaly_by_month.png", dpi=150)
plt.close()

print("  ✓ All 6 charts saved to results/")


# ─────────────────────────────────────────────
# 8. BUSINESS INSIGHTS
# ─────────────────────────────────────────────
print("\n💼 Business Impact Analysis")
print("="*55)

COST_PER_KWH = 0.12   # $0.12 / kWh — adjust for your region

anomaly_df = df[df["is_anomaly"] == 1].copy()
normal_df  = df[df["is_anomaly"] == 0].copy()

# ── 8a: Estimated cost impact ──
avg_normal_elec  = normal_df["electricity"].mean()
excess_per_event = (anomaly_df["electricity"] - avg_normal_elec).clip(lower=0)
total_excess_kwh = excess_per_event.sum()
cost_impact      = total_excess_kwh * COST_PER_KWH

print(f"\n💰 Estimated excess energy from anomalies")
print(f"   Excess kWh    : {total_excess_kwh:,.0f} kWh")
print(f"   Cost impact   : ${cost_impact:,.0f}")

# ── 8b: Seasonal pattern ──
seasonal = anomaly_df.groupby("month").size()
peak_month = seasonal.idxmax()
print(f"\n📅 Seasonal Patterns")
print(f"   Peak anomaly month : {month_labels[int(peak_month)-1]}")
print(f"   Anomalies by month :\n{seasonal.to_string()}")

# ── 8c: Peak hours ──
peak_hours = anomaly_df.groupby("hour").size().sort_values(ascending=False)
print(f"\n⏰ Top-5 Peak Anomaly Hours")
print(peak_hours.head(5).to_string())

# ── 8d: Anomaly type classification ──
if "electricity_dev" in anomaly_df.columns:
    anomaly_df["anomaly_type"] = pd.cut(
        anomaly_df["electricity_dev"],
        bins=[-np.inf, -1.5, 1.5, np.inf],
        labels=["Drop (Under-consumption)", "Normal Range", "Spike (Over-consumption)"]
    )
    print(f"\n🔖 Anomaly Type Breakdown")
    print(anomaly_df["anomaly_type"].value_counts().to_string())

# ── 8e: Per-building summary (if available) ──
if "building_id" in df.columns:
    building_summary = (
        df.groupby("building_id")
        .agg(
            total_records=("is_anomaly", "count"),
            anomaly_count=("is_anomaly", "sum"),
            avg_electricity=("electricity", "mean"),
        )
        .assign(anomaly_rate=lambda x: x["anomaly_count"] / x["total_records"] * 100)
        .sort_values("anomaly_rate", ascending=False)
        .round(2)
    )
    print(f"\n🏢 Per-Building Anomaly Summary")
    print(building_summary.to_string())
    building_summary.to_csv("results/building_summary.csv")

# ── 8f: Business recommendations ──
print(f"""
{'='*55}
📋 ACTIONABLE RECOMMENDATIONS
{'='*55}
1. IMMEDIATE INSPECTION
   Buildings with anomaly rate > 10% need immediate
   engineering review for equipment faults.

2. SCHEDULING OPTIMIZATION
   Hour {int(peak_hours.index[0]):02d}:00 shows highest anomaly rate.
   Review HVAC scheduling and occupancy patterns.

3. SEASONAL MAINTENANCE
   {month_labels[int(peak_month)-1]} shows peak anomalies — schedule preventive
   maintenance before this period each year.

4. COST RECOVERY TARGET
   Addressing top anomalies could save ≈ ${cost_impact*0.4:,.0f}
   (assuming 40% correctable via operations).

5. MODEL DEPLOYMENT
   Retrain Isolation Forest monthly as building patterns
   shift. Alert threshold: anomaly_score > 0.6.
{'='*55}
""")


# ─────────────────────────────────────────────
# 9. EXPORT RESULTS
# ─────────────────────────────────────────────
output_cols = [
    "timestamp", "building_id", "electricity", "temperature",
    "is_anomaly", "anomaly_votes", "anomaly_score",
    "vote_iso", "vote_lof", "vote_robust"
] if "building_id" in df.columns else [
    "timestamp", "electricity", "temperature",
    "is_anomaly", "anomaly_votes", "anomaly_score"
]
output_cols = [c for c in output_cols if c in df.columns]

df[output_cols].to_csv("results/anomaly_predictions.csv", index=False)
print(f"✅ Predictions exported → results/anomaly_predictions.csv")
print(f"\n🎉 Pipeline complete! Check results/ for all outputs.")