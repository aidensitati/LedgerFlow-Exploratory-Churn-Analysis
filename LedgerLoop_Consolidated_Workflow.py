# The Sensitivity of High-Tier Clients in SaaS Products
# Analyzing how structural intake conditions shape churn risk in enterprise SaaS customers.

### Stage 1: Dataset Consolidation
# Consolidated customers, subscriptions, and revenue into a single canonical table.
# Normalized temporal data to monthly resolution and computed churn and tenure.

import pandas as pd
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
BASE_PATH = Path(r"C:\Users\hp\Exploratory Data Analysis\CSV files")

CUSTOMERS_PATH = BASE_PATH / "customers.csv"
REVENUE_PATH = BASE_PATH / "revenue.csv"
SUBSCRIPTIONS_PATH = BASE_PATH / "subscriptions.csv"

OUTPUT_PATH = BASE_PATH / "ledgerloop_consolidated.csv"

# -----------------------------
# Load datasets
# -----------------------------
customers = pd.read_csv(CUSTOMERS_PATH, parse_dates=["signup_date", "churn_date"])
revenue = pd.read_csv(REVENUE_PATH, parse_dates=["month"])
subscriptions = pd.read_csv(SUBSCRIPTIONS_PATH, parse_dates=["month"])

# -----------------------------
# Normalize time (monthly)
# -----------------------------
revenue["month"] = revenue["month"].dt.to_period("M").dt.to_timestamp()
subscriptions["month"] = subscriptions["month"].dt.to_period("M").dt.to_timestamp()

# -----------------------------
# Revenue aggregation (mechanical only)
# -----------------------------
revenue_agg = (
    revenue
    .groupby("customer_id", as_index=False)
    .agg(
        total_revenue=("amount", "sum"),
        avg_monthly_revenue=("amount", "mean"),
        revenue_month_count=("month", "nunique")
    )
)

# -----------------------------
# Subscription aggregation (structural only)
# -----------------------------
subscription_agg = (
    subscriptions
    .groupby("customer_id", as_index=False)
    .agg(
        subscription_months=("month", "nunique"),
        avg_list_price=("monthly_fee", "mean")
    )
)

# -----------------------------
# Churn construction (terminal, binary)
# -----------------------------
customers["churn_flag"] = customers["churn_date"].notna().astype(int)

# -----------------------------
# Tenure construction (lifecycle descriptor)
# -----------------------------
reference_date = customers["churn_date"].fillna(pd.Timestamp.today())

customers["tenure_months"] = (
    (reference_date.dt.to_period("M") - customers["signup_date"].dt.to_period("M"))
    .apply(lambda p: p.n)
)

# -----------------------------
# Assemble final canonical table
# -----------------------------
final = (
    customers
    .merge(revenue_agg, on="customer_id", how="left")
    .merge(subscription_agg, on="customer_id", how="left")
)

# -----------------------------
# Coverage flag (lifecycle completeness)
# -----------------------------
final["coverage_flag"] = final["churn_date"].notna().astype(int)

# -----------------------------
# Fill mechanically valid nulls
# -----------------------------
final[[
    "total_revenue",
    "avg_monthly_revenue",
    "revenue_month_count",
    "subscription_months",
    "avg_list_price"
]] = final[[
    "total_revenue",
    "avg_monthly_revenue",
    "revenue_month_count",
    "subscription_months",
    "avg_list_price"
]].fillna(0)

# -----------------------------
# Final sanity checks
# -----------------------------
assert final["customer_id"].is_unique, "Row duplication detected"
assert "churn_flag" in final.columns, "churn_flag missing"
assert "tenure_months" in final.columns, "tenure_months missing"
assert "coverage_flag" in final.columns, "coverage_flag missing"

# -----------------------------
# Save output
# -----------------------------
final.to_csv(OUTPUT_PATH, index=False)

print(f"Consolidated dataset saved to: {OUTPUT_PATH}")




# ===============================

### Stage 2: Data Integrity & Leakage Risk
# Verified dataset integrity, identified structural leakage risks, excluded post-outcome contaminated variables.

import pandas as pd
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
DATA_PATH = Path(r"C:\Users\hp\Exploratory Data Analysis\CSV files\ledgerloop_consolidated.csv")
OUTPUT_PATH = DATA_PATH.with_name(
    DATA_PATH.stem + " (Leakage risk neutralized, data integrity maintained).csv"
)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["signup_date", "churn_date"])

print("\n--- DATA LOADED ---")
print(f"Shape: {df.shape}")

# -----------------------------
# Schema overview
# -----------------------------
schema = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "null_rate": df.isna().mean()
})

print("\n--- SCHEMA OVERVIEW ---")
print(schema)

# -----------------------------
# Churn label integrity checks
# -----------------------------
print("\n--- CHURN CONSISTENCY CHECKS ---")

inconsistent_churn = df[
    ((df["churn_flag"] == 1) & (df["churn_date"].isna())) |
    ((df["churn_flag"] == 0) & (df["churn_date"].notna()))
]

print(f"Inconsistent churn records: {len(inconsistent_churn)}")

# -----------------------------
# Coverage enforcement
# -----------------------------
print("\n--- COVERAGE CHECK ---")

coverage_counts = df["coverage_flag"].value_counts(dropna=False)
print(coverage_counts)

eligible_df = df[df["coverage_flag"] == 1].copy()

print(f"Records eligible for EDA (coverage_flag == 1): {eligible_df.shape[0]}")

# -----------------------------
# Temporal sanity checks
# -----------------------------
print("\n--- TEMPORAL SANITY CHECKS ---")

invalid_temporal = eligible_df[
    (eligible_df["churn_date"].notna()) &
    (eligible_df["churn_date"] < eligible_df["signup_date"])
]

print(f"Records with churn before signup: {len(invalid_temporal)}")

# -----------------------------
# Leakage risk classification
# -----------------------------
LEAKAGE_HIGH_RISK = [
    "churn_date",
    "tenure_months",
    "total_revenue",
    "avg_monthly_revenue",
    "revenue_month_count",
    "subscription_months"
]

LEAKAGE_MEDIUM_RISK = [
    "avg_list_price"
]

SAFE_FOR_EDA = [
    "customer_id",
    "signup_date",
    "plan_type",
    "monthly_fee",
    "acquisition_cost",
    "coverage_flag",
    "churn_flag"
]

print("\n--- LEAKAGE RISK MAP ---")
print("High risk (excluded from EDA):", LEAKAGE_HIGH_RISK)
print("Medium risk (context only):", LEAKAGE_MEDIUM_RISK)
print("Safe for EDA:", SAFE_FOR_EDA)

# -----------------------------
# Construct leakage-neutral EDA dataset
# -----------------------------
eda_df = eligible_df[SAFE_FOR_EDA + LEAKAGE_MEDIUM_RISK].copy()

# Explicitly annotate exclusions
eda_metadata = pd.DataFrame({
    "column": df.columns,
    "included_in_eda": df.columns.isin(eda_df.columns),
    "leakage_risk": [
        "HIGH" if c in LEAKAGE_HIGH_RISK else
        "MEDIUM" if c in LEAKAGE_MEDIUM_RISK else
        "LOW"
        for c in df.columns
    ]
})

print("\n--- EDA COLUMN ELIGIBILITY ---")
print(eda_metadata)

# -----------------------------
# Save leakage-neutral dataset
# -----------------------------
eda_df.to_csv(OUTPUT_PATH, index=False)

print("\n--- OUTPUT SAVED ---")
print(f"File written to:\n{OUTPUT_PATH}")

# -----------------------------
# Final integrity summary
# -----------------------------
print("\n--- FINAL INTEGRITY SUMMARY ---")
print(f"Original rows: {df.shape[0]}")
print(f"EDA-eligible rows: {eda_df.shape[0]}")
print(f"Columns excluded due to leakage risk: {len(LEAKAGE_HIGH_RISK)}")
print("No values imputed. No labels modified. No post-outcome data used.")


# ===============================

### Stage 3: Target Variable Analysis
# Confirmed churn_flag as administrative, examined tenure, plan types, and coverage flag to understand structural churn patterns.

import pandas as pd
import numpy as np

# -----------------------------
# Load dataset
# -----------------------------
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\ledgerloop_consolidated.csv"
df = pd.read_csv(file_path, parse_dates=['signup_date', 'churn_date'])

# -----------------------------
# Basic target prevalence (numerical only)
# -----------------------------
total_customers = df.shape[0]
observed_churn = df['churn_flag'].sum()
observed_active = total_customers - observed_churn
overall_churn_rate = observed_churn / total_customers

print("=== TARGET PREVALENCE ===")
print(f"Total customers: {total_customers}")
print(f"Observed churn events: {observed_churn}")
print(f"Observed non-churn (administrative label, right-censored): {observed_active}")
print(f"Overall observed churn proportion: {overall_churn_rate:.4f}\n")

# -----------------------------
# Observation window distribution (tenure-based)
# -----------------------------
tenure_summary = df['tenure_months'].describe()
print("=== OBSERVATION WINDOWS (TENURE MONTHS) ===")
print(tenure_summary, "\n")

# -----------------------------
# Segment-level churn (descriptive only)
# -----------------------------
segment_churn = df.groupby('plan_type').agg(
    total_customers=('customer_id', 'count'),
    observed_churn=('churn_flag', 'sum'),
    churn_proportion=('churn_flag', 'mean'),
    median_tenure=('tenure_months', 'median')
)
print("=== CHURN BY PLAN TYPE (OBSERVED) ===")
print(segment_churn, "\n")

coverage_churn = df.groupby('coverage_flag').agg(
    total_customers=('customer_id', 'count'),
    observed_churn=('churn_flag', 'sum'),
    churn_proportion=('churn_flag', 'mean'),
    median_tenure=('tenure_months', 'median')
)
print("=== CHURN BY COVERAGE FLAG (OBSERVED) ===")
print(coverage_churn, "\n")

# -----------------------------
# Tenure-based churn behavior
# -----------------------------
# Buckets for descriptive temporal framing
tenure_bins = [0, 3, 6, 12, 24, np.inf]
tenure_labels = ['0-3', '3-6', '6-12', '12-24', '24+']
df['tenure_bucket'] = pd.cut(df['tenure_months'], bins=tenure_bins, labels=tenure_labels, right=False)

tenure_churn = df.groupby('tenure_bucket').agg(
    customers_in_bucket=('customer_id', 'count'),
    observed_churn=('churn_flag', 'sum'),
    churn_proportion=('churn_flag', 'mean')
)
print("=== CHURN BY TENURE BUCKET (OBSERVED) ===")
print(tenure_churn, "\n")

# -----------------------------
# Numeric features descriptive comparison
# -----------------------------
numeric_features = [
    'monthly_fee', 'acquisition_cost', 'tenure_months',
    'total_revenue', 'avg_monthly_revenue', 'avg_list_price', 'subscription_months'
]

behavioral_summary = df.groupby('churn_flag')[numeric_features].agg(['mean', 'median'])
print("=== NUMERIC FEATURES BY CHURN FLAG ===")
print(behavioral_summary, "\n")

# -----------------------------
# Standardized mean differences (numerical only)
# -----------------------------
smds = {}
for feature in numeric_features:
    mean_churn = df.loc[df['churn_flag'] == 1, feature].mean()
    mean_active = df.loc[df['churn_flag'] == 0, feature].mean()
    std_active = df.loc[df['churn_flag'] == 0, feature].std()
    smds[feature] = (mean_churn - mean_active) / std_active

print("=== STANDARDIZED MEAN DIFFERENCES (CHURN - ACTIVE) ===")
for feature, smd in smds.items():
    print(f"{feature}: {smd:.3f}")

# -----------------------------
# Baseline anchors (naive and heuristic)
# -----------------------------
# Naive baseline: predict "no observed churn"
naive_accuracy = observed_active / total_customers

# Heuristic baseline: customers with tenure below median are flagged
median_tenure = df['tenure_months'].median()
df['heuristic_churn'] = (df['tenure_months'] < median_tenure).astype(int)
heuristic_accuracy = (df['heuristic_churn'] == df['churn_flag']).mean()
heuristic_delta = heuristic_accuracy - naive_accuracy

print("\n=== BASELINE ANCHORS ===")
print(f"Naive baseline (all non-churn observed): {naive_accuracy:.4f}")
print(f"Heuristic rule: tenure < median ({median_tenure} months)")
print(f"Heuristic accuracy: {heuristic_accuracy:.4f}")
print(f"Improvement over naive baseline: {heuristic_delta:.4f}")

# -----------------------------
# Structural comparison: completed vs ongoing lifecycles
# (Pre-churn structural characteristics)
# -----------------------------

print("\n=== STRUCTURAL DIFFERENCES: COMPLETED vs ONGOING LIFECYCLES ===")

# Define lifecycle status
df['lifecycle_status'] = np.where(df['churn_flag'] == 1, 'completed', 'ongoing')

# -----------------------------
# Tenure exposure comparison
# -----------------------------
tenure_exposure = df.groupby('lifecycle_status')['tenure_months'].describe()
print("\n--- TENURE EXPOSURE COMPARISON ---")
print(tenure_exposure)

# -----------------------------
# Categorical structural composition
# -----------------------------
plan_distribution = (
    df.groupby(['lifecycle_status', 'plan_type'])
      .size()
      .unstack(fill_value=0)
      .div(df.groupby('lifecycle_status').size(), axis=0)
)

print("\n--- PLAN TYPE DISTRIBUTION (PROPORTIONS) ---")
print(plan_distribution)

coverage_distribution = (
    df.groupby(['lifecycle_status', 'coverage_flag'])
      .size()
      .unstack(fill_value=0)
      .div(df.groupby('lifecycle_status').size(), axis=0)
)

print("\n--- COVERAGE FLAG DISTRIBUTION (PROPORTIONS) ---")
print(coverage_distribution)

# -----------------------------
# Numeric structural summaries (pre-churn)
# -----------------------------
structural_numeric_summary = df.groupby('lifecycle_status')[numeric_features].agg(
    ['mean', 'median', 'std']
)

print("\n--- NUMERIC STRUCTURAL SUMMARY ---")
print(structural_numeric_summary)

# -----------------------------
# Standardized mean differences
# (Completed lifecycle relative to ongoing)
# -----------------------------
structural_smds = {}

for feature in numeric_features:
    mean_completed = df.loc[df['lifecycle_status'] == 'completed', feature].mean()
    mean_ongoing = df.loc[df['lifecycle_status'] == 'ongoing', feature].mean()
    std_ongoing = df.loc[df['lifecycle_status'] == 'ongoing', feature].std()
    structural_smds[feature] = (mean_completed - mean_ongoing) / std_ongoing

print("\n--- STRUCTURAL STANDARDIZED MEAN DIFFERENCES (COMPLETED - ONGOING) ---")
for feature, smd in structural_smds.items():
    print(f"{feature}: {smd:.3f}")

# -----------------------------
# Exposure-normalized churn concentration
# -----------------------------
exposure_adjusted = df.groupby('lifecycle_status').agg(
    customers=('customer_id', 'count'),
    total_tenure=('tenure_months', 'sum'),
)

exposure_adjusted['customers_per_tenure_month'] = (
    exposure_adjusted['customers'] / exposure_adjusted['total_tenure']
)

print("\n--- EXPOSURE NORMALIZED PRESENCE ---")
print(exposure_adjusted)


# ===============================

### Stage 4: Feature Triage
# Produced a minimal high-integrity feature space to map early lifecycle structural conditions leading to churn.

import pandas as pd

# ----------------------------
# Paths
# ----------------------------
base_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files"

original_path = base_path + r"\ledgerloop_consolidated.csv"
leakage_neutral_path = base_path + r"\ledgerloop_consolidated (Leakage risk neutralized, data integrity maintained).csv"

retained_safe_output_path = base_path + r"\ledgerloop_retained_leakage_free.csv"
triage_output_path = base_path + r"\ledgerloop_feature_triage.csv"

# ----------------------------
# Columns explicitly removed due to leakage
# ----------------------------
leakage_columns = [
    "churn_date",
    "tenure_months",
    "total_revenue",
    "avg_monthly_revenue",
    "revenue_month_count",
    "subscription_months"
]

# ----------------------------
# Load original dataset
# ----------------------------
df = pd.read_csv(original_path)

# ----------------------------
# Create leakage-free dataset WITH retained customers
# ----------------------------
df_retained_safe = df.drop(columns=leakage_columns, errors="ignore")

df_retained_safe.to_csv(retained_safe_output_path, index=False)

print("Leakage-free dataset with retained customers saved:")
print(retained_safe_output_path)
print()

# ----------------------------
# Feature triage (LedgerLoop rules)
# ----------------------------

triage_records = []

for feature in df_retained_safe.columns:

    # --- Exclude identifiers from triage ---
    if feature == "customer_id":
        triage_records.append({
            "feature": feature,
            "temporal_admissibility": "pre-decision admissible",
            "exploratory_priority": "discard",
            "exposure_dependency": "no"
        })
        continue

    # --- Context-only feature ---
    if feature == "avg_list_price":
        triage_records.append({
            "feature": feature,
            "temporal_admissibility": "pre-decision admissible",
            "exploratory_priority": "secondary",
            "exposure_dependency": "no"
        })
        continue

    # --- Date features ---
    if feature == "signup_date":
        triage_records.append({
            "feature": feature,
            "temporal_admissibility": "pre-decision admissible",
            "exploratory_priority": "secondary",
            "exposure_dependency": "indirect (cohort proxy)"
        })
        continue

    # --- Core structural early-life features ---
    if feature in ["plan_type", "monthly_fee", "acquisition_cost"]:
        triage_records.append({
            "feature": feature,
            "temporal_admissibility": "pre-decision admissible",
            "exploratory_priority": "high-priority",
            "exposure_dependency": "no"
        })
        continue

    # --- Fallback (should be rare) ---
    triage_records.append({
        "feature": feature,
        "temporal_admissibility": "ambiguous",
        "exploratory_priority": "discard",
        "exposure_dependency": "unknown"
    })

# ----------------------------
# Create triage DataFrame
# ----------------------------
triage_df = pd.DataFrame(triage_records).drop_duplicates(subset=["feature"])

# ----------------------------
# Save triage output
# ----------------------------
triage_df.to_csv(triage_output_path, index=False)

# ----------------------------
# Display triage results
# ----------------------------
print("LedgerLoop Feature Triage Output:")
print(triage_df)
print()
print("Feature triage saved to:")
print(triage_output_path)


# ===============================

### Stage 5: Diagnostic Exploration
# Parsed temporal, numerical, and categorical features; identified February as a multi-variable anomalous cohort with structural shifts preceding churn.

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\ledgerloop_consolidated (Leakage risk neutralized, data integrity maintained).csv"
df = pd.read_csv(file_path, parse_dates=['signup_date'])

# Step 1 — Feature Selection
high_priority_features = ['plan_type', 'monthly_fee', 'acquisition_cost']
context_features = ['signup_date', 'avg_list_price']
exclude_features = ['churn_flag', 'coverage_flag', 'customer_id']  # Exclude IDs and post-outcome

analysis_features = high_priority_features + context_features
df_analysis = df[analysis_features].copy()

# Step 2: Distributional Stress Analysis
high_priority_features = ['plan_type', 'monthly_fee', 'acquisition_cost']

distribution_summary = {}

for feature in high_priority_features:
    if not pd.api.types.is_numeric_dtype(df_analysis[feature]):
        continue  # Skip non-numeric just in case
    
    data = df_analysis[feature]
    
    summary = {
        'count': data.count(),
        'mean': data.mean(),
        'median': data.median(),
        'min': data.min(),
        'max': data.max(),
        '25%': data.quantile(0.25),
        '75%': data.quantile(0.75),
        'variance': data.var(),
        'std_dev': data.std()
    }
    
    distribution_summary[feature] = summary

# Convert to DataFrame for nice display
distribution_df = pd.DataFrame(distribution_summary).T
print("\n=== Distributional Summary (Step 2) ===")
print(distribution_df)

# Step 3 — Exposure-Aligned Trajectories (Numerical)

# Cohort grouping by signup month to simulate exposure alignment
df_analysis['signup_month'] = df_analysis['signup_date'].dt.to_period('M')

# Separate numeric and categorical high-priority features
high_priority_numeric = ['monthly_fee', 'acquisition_cost']
high_priority_categorical = ['plan_type']  # Keep for reference, not numeric aggregation

# Aggregate numeric features only
cohort_summary_numeric = df_analysis.groupby('signup_month')[high_priority_numeric].agg(['mean', 'std', 'median'])

# Flatten MultiIndex columns
cohort_summary_numeric.columns = ['_'.join(col) for col in cohort_summary_numeric.columns]

print("\n=== Cohort-wise Exposure-Aligned Summary (Numeric Features) ===")
print(cohort_summary_numeric.head())

# Optional: summarize categorical features per cohort (frequency counts)
cohort_summary_categorical = df_analysis.groupby('signup_month')[high_priority_categorical].agg(lambda x: x.value_counts().to_dict())
print("\n=== Cohort-wise Exposure-Aligned Summary (Categorical Features) ===")
print(cohort_summary_categorical.head())

# Step 4 — Multivariate Structure Analysis (Numerical)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Only numeric high-priority features
high_priority_numeric = ['monthly_fee', 'acquisition_cost']

# Standardize numeric features for PCA / clustering
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df_analysis[high_priority_numeric])

# PCA
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(X_numeric)
df_analysis['pca1'] = pca_components[:, 0]
df_analysis['pca2'] = pca_components[:, 1]

print("\n=== PCA Explained Variance Ratio ===")
print(pca.explained_variance_ratio_)

# KMeans clustering to identify structural regimes
kmeans = KMeans(n_clusters=3, random_state=42)
df_analysis['cluster'] = kmeans.fit_predict(X_numeric)

# Cluster summary (numeric features only)
cluster_summary = df_analysis.groupby('cluster')[high_priority_numeric].agg(['mean', 'std', 'min', 'max'])
cluster_summary.columns = ['_'.join(col) for col in cluster_summary.columns]

print("\n=== Cluster Structural Summary ===")
print(cluster_summary)


# Step 5 — Bottleneck Flagging
# Simple heuristic: bottleneck if numeric feature exceeds 75th percentile or falls below 25th percentile in cluster

# Only numeric high-priority features
numeric_features = ['monthly_fee', 'acquisition_cost']

bottleneck_flags = pd.DataFrame(columns=['feature', 'cluster', 'low_bottleneck', 'high_bottleneck'])

for feature in numeric_features:
    for cluster in df_analysis['cluster'].unique():
        cluster_data = df_analysis[df_analysis['cluster'] == cluster][feature]
        # Compute flags based on percentiles within the cluster
        low_flag = (cluster_data < cluster_data.quantile(0.25)).any()
        high_flag = (cluster_data > cluster_data.quantile(0.75)).any()
        
        # Append to the bottleneck_flags DataFrame
        bottleneck_flags = pd.concat([
            bottleneck_flags, 
            pd.DataFrame({
                'feature': [feature],
                'cluster': [cluster],
                'low_bottleneck': [low_flag],
                'high_bottleneck': [high_flag]
            })
        ], ignore_index=True)

print("\n=== Bottleneck Flags by Feature and Cluster ===")
print(bottleneck_flags)
# Step 6 — Documentation / Save Results
output_path_base = r"C:\Users\hp\Exploratory Data Analysis\CSV files\ledgerloop_diagnostic_exploration"

# Save main analysis-ready dataset
df_analysis.to_csv(output_path_base + "_analysis_ready.csv", index=False)

# Combine distribution summaries into a single DataFrame for saving
combined_distribution_summary = pd.concat(
    [pd.DataFrame.from_dict(summary, orient='index', columns=['value']).reset_index().rename(columns={'index': 'statistic'}).assign(feature=feat)
     for feat, summary in distribution_summary.items()],
    ignore_index=True
)
combined_distribution_summary.to_csv(output_path_base + "_distribution_summary.csv", index=False)

# Save cohort summaries if they exist
if 'cohort_summary_categorical' in globals():
    cohort_summary_categorical.to_csv(output_path_base + "_cohort_summary_categorical.csv", index=False)
if 'cohort_summary_numeric' in globals():
    cohort_summary_numeric.to_csv(output_path_base + "_cohort_summary_numeric.csv", index=False)

# Save cluster summary
cluster_summary.to_csv(output_path_base + "_cluster_summary.csv", index=False)

# Save bottleneck flags
bottleneck_flags.to_csv(output_path_base + "_bottleneck_flags.csv", index=False)

print("\nAll diagnostic exploration outputs saved successfully.")

import os

# --- Step 5c — Combined Numeric + Categorical Bottleneck Scoring ---

# Identify numeric and categorical features
numeric_features = df_analysis.select_dtypes(include='number').columns.tolist()
categorical_features = df_analysis.select_dtypes(include='object').columns.tolist()

# Initialize the combined bottleneck scores DataFrame
combined_bottleneck_scores = pd.DataFrame(columns=[
    'feature', 'cluster', 'score', 'extremes', 'std_contrib', 'escalation', 'categorical_bottleneck'
])

# Loop through clusters
for cluster in df_analysis['cluster'].unique():
    cluster_df = df_analysis[df_analysis['cluster'] == cluster]

    # Numeric features
    for feature in numeric_features:
        cluster_data = cluster_df[feature]

        q25, q75 = cluster_data.quantile([0.25, 0.75])
        extremes = ((cluster_data < q25) | (cluster_data > q75)).sum() / len(cluster_data)
        cluster_std = cluster_data.std()
        max_std = df_analysis[numeric_features].std().max()
        std_contrib = cluster_std / (max_std + 1e-6)

        cohort_means = cluster_df.groupby(cluster_df['signup_date'].dt.to_period('M'))[feature].mean().sort_index()
        early_months = cohort_means.iloc[:3]
        escalation = (early_months.iloc[-1] - early_months.iloc[0]) / (early_months.iloc[0] + 1e-6) if len(early_months) >= 2 else 0

        score_numeric = (extremes + std_contrib + abs(escalation)) / 3

        combined_bottleneck_scores = pd.concat([
            combined_bottleneck_scores,
            pd.DataFrame({
                'feature': [feature],
                'cluster': [cluster],
                'score': [score_numeric],
                'extremes': [extremes],
                'std_contrib': [std_contrib],
                'escalation': [escalation],
                'categorical_bottleneck': [0]  # numeric features have no categorical contribution
            })
        ], ignore_index=True)

    # Categorical features
    for feature in categorical_features:
        counts = cluster_df[feature].value_counts(normalize=True)
        dominant = counts.max()
        rare = counts.min()

        cat_score = max(0, dominant - 0.75) + max(0, 0.1 - rare)
        cat_score = min(cat_score, 1)

        combined_bottleneck_scores = pd.concat([
            combined_bottleneck_scores,
            pd.DataFrame({
                'feature': [feature],
                'cluster': [cluster],
                'score': [cat_score],
                'extremes': [0],
                'std_contrib': [0],
                'escalation': [0],
                'categorical_bottleneck': [cat_score]
            })
        ], ignore_index=True)

# Sort by overall score
combined_bottleneck_scores.sort_values(by='score', ascending=False, inplace=True)

# Ensure folder exists
output_folder = r"C:\Users\hp\Exploratory Data Analysis\CSV files"
os.makedirs(output_folder, exist_ok=True)
output_path_base = os.path.join(output_folder, "ledgerloop_diagnostic_exploration")

# Save CSV
combined_bottleneck_scores.to_csv(output_path_base + "_combined_bottleneck_scores.csv", index=False)

print(f"\nCombined numeric + categorical bottleneck scoring completed and saved at:\n{output_path_base}_combined_bottleneck_scores.csv")


# ===============================

### Stage 6: Diagnostic Exploration – Bottleneck Analysis
# Applied numeric and categorical bottleneck scoring to identify concentrated structural risk points.

import pandas as pd
import numpy as np
from datetime import datetime

# --- Step 0: Load dataset ---
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\ledgerloop_consolidated (Leakage risk neutralized, data integrity maintained).csv"
df = pd.read_csv(file_path, parse_dates=['signup_date'])

# --- Step 1: Create cohort (signup_month) ---
df['signup_month'] = df['signup_date'].dt.to_period('M')

numeric_features = ['monthly_fee', 'acquisition_cost', 'avg_list_price']
categorical_features = ['plan_type', 'coverage_flag', 'churn_flag']

# --- Step 2: Cohort-level numeric summaries ---
cohort_numeric_summary = df.groupby('signup_month')[numeric_features].agg(['mean','median','std','min','max'])
cohort_numeric_summary.columns = ['_'.join(col) for col in cohort_numeric_summary.columns]

# --- Step 3: Cohort-level categorical summaries ---
def categorical_counts(series):
    return series.value_counts().to_dict()

cohort_categorical_summary = df.groupby('signup_month')[categorical_features].agg(categorical_counts)

# --- Step 4: Identify skewed categorical features for February 2024 ---
target_cohort = '2024-02'
feb_df = df[df['signup_month'] == target_cohort]

categorical_skew = {}
for feature in categorical_features:
    counts = feb_df[feature].value_counts(normalize=True)
    skewed = counts[counts > 0.75].to_dict()  # Dominant category > 75%
    categorical_skew[feature] = skewed

# --- Step 5: Compare numeric features of Feb vs other months ---
comparison = pd.DataFrame()
for feature in numeric_features:
    feb_mean = feb_df[feature].mean()
    overall_mean = df[feature].mean()
    comparison.loc[feature, 'feb_mean'] = feb_mean
    comparison.loc[feature, 'overall_mean'] = overall_mean
    comparison.loc[feature, 'diff'] = feb_mean - overall_mean

# --- Step 6: Lifecycle variance (monthly progression for Feb cohort) ---
lifecycle_variance = {}
for feature in numeric_features:
    monthly_means = feb_df.groupby(feb_df['signup_date'].dt.to_period('M'))[feature].mean()
    if len(monthly_means) >= 2:
        lifecycle_variance[feature] = monthly_means.diff().abs().sum()
    else:
        lifecycle_variance[feature] = 0

# --- Step 7: Combined bottleneck scoring ---
combined_scores = []

for feature in numeric_features:
    extremes = ((feb_df[feature] < feb_df[feature].quantile(0.25)) | 
                (feb_df[feature] > feb_df[feature].quantile(0.75))).mean()
    std_contrib = feb_df[feature].std() / (df[feature].std() + 1e-6)
    escalation = lifecycle_variance[feature] / (df[feature].std() + 1e-6)
    score = (extremes + std_contrib + escalation) / 3
    combined_scores.append({
        'feature': feature,
        'cohort': target_cohort,
        'numeric_score': score,
        'extremes': extremes,
        'std_contrib': std_contrib,
        'escalation': escalation,
        'categorical_score': 0
    })

for feature, skewed_categories in categorical_skew.items():
    if skewed_categories:
        cat_score = sum([v for v in skewed_categories.values()])  # simple sum as score
    else:
        cat_score = 0
    combined_scores.append({
        'feature': feature,
        'cohort': target_cohort,
        'numeric_score': 0,
        'extremes': 0,
        'std_contrib': 0,
        'escalation': 0,
        'categorical_score': cat_score
    })

combined_scores_df = pd.DataFrame(combined_scores)
combined_scores_df['total_score'] = combined_scores_df['numeric_score'] + combined_scores_df['categorical_score']
combined_scores_df.sort_values('total_score', ascending=False, inplace=True)

# --- Step 8: Save outputs ---
output_base = r"C:\Users\hp\Exploratory Data Analysis\CSV files\feb_cohort_bottleneck_analysis"

df.to_csv(output_base + "_full_dataset.csv", index=False)
cohort_numeric_summary.to_csv(output_base + "_cohort_numeric_summary.csv")
cohort_categorical_summary.to_csv(output_base + "_cohort_categorical_summary.csv")
comparison.to_csv(output_base + "_feb_vs_overall_numeric_comparison.csv")
combined_scores_df.to_csv(output_base + "_combined_bottleneck_scores.csv", index=False)

print("Analysis completed. All CSVs saved.")
print("\n=== Combined Bottleneck Scores ===")
print(combined_scores_df)


# ===============================

### Stage 7: Inferential Exploration
# Analyzed retained vs churned cohorts; identified enterprise customers with high pricing as a structurally vulnerable group.
# Confirmed intake configurations, temporal admissibility, and limited generalization.
# Summary: Churn at LedgerLoop is largely shaped at intake, with elevated risk concentrated in enterprise customers exposed to higher pricing structures; February surfaced this risk due to disproportionate onboarding of such configurations.

import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# Step 0 — Load data
# ===============================
file_path = r"C:\Users\hp\Exploratory Data Analysis\CSV files\ledgerloop_retained_leakage_free.csv"
df = pd.read_csv(file_path, parse_dates=['signup_date'])

output_dir = Path(file_path).parent

# Core intake features
numeric_features = ['monthly_fee', 'acquisition_cost', 'avg_list_price']
categorical_features = ['plan_type', 'coverage_flag']
target = 'churn_flag'

# Create cohort
df['signup_month'] = df['signup_date'].dt.to_period('M')

# February cohort
feb = df[df['signup_month'] == '2024-02']

# ===============================
# Phase 1 — February retained vs February churned
# ===============================
feb_retained = feb[feb[target] == 0]
feb_churned = feb[feb[target] == 1]

phase1_numeric = (
    feb.groupby(target)[numeric_features]
       .agg(['mean', 'median', 'std', 'min', 'max'])
)
phase1_numeric.columns = ['_'.join(col) for col in phase1_numeric.columns]
phase1_numeric.reset_index(inplace=True)

phase1_categorical = (
    feb.groupby(target)[categorical_features]
       .agg(lambda x: x.value_counts().to_dict())
       .reset_index()
)

phase1_numeric.to_csv(output_dir / "phase1_feb_retained_vs_churned_numeric.csv", index=False)
phase1_categorical.to_csv(output_dir / "phase1_feb_retained_vs_churned_categorical.csv", index=False)

print("\n=== Phase 1: February Retained vs Churned — Numeric ===")
print(phase1_numeric)

print("\n=== Phase 1: February Retained vs Churned — Categorical ===")
print(phase1_categorical)

# ===============================
# Phase 2 — February retained vs other months retained
# ===============================
retained = df[df[target] == 0]

phase2_numeric = (
    retained.groupby('signup_month')[numeric_features]
            .agg(['mean', 'median', 'std'])
)
phase2_numeric.columns = ['_'.join(col) for col in phase2_numeric.columns]
phase2_numeric.reset_index(inplace=True)

phase2_categorical = (
    retained.groupby('signup_month')['plan_type']
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
)

phase2_numeric.to_csv(output_dir / "phase2_retained_by_cohort_numeric.csv", index=False)
phase2_categorical.to_csv(output_dir / "phase2_retained_by_cohort_plan_distribution.csv", index=False)

print("\n=== Phase 2: Retained Customers — Numeric by Cohort ===")
print(phase2_numeric)

print("\n=== Phase 2: Retained Customers — Plan Distribution by Cohort ===")
print(phase2_categorical)

# ===============================
# Phase 3 — Enterprise vs Non-Enterprise (Churned & Retained)
# ===============================
df['is_enterprise'] = (df['plan_type'] == 'Enterprise').astype(int)

phase3 = (
    df.groupby([target, 'is_enterprise'])[numeric_features]
      .agg(['mean', 'median', 'std'])
)
phase3.columns = ['_'.join(col) for col in phase3.columns]
phase3.reset_index(inplace=True)

phase3.to_csv(output_dir / "phase3_enterprise_vs_nonenterprise_structural.csv", index=False)

print("\n=== Phase 3: Enterprise vs Non-Enterprise Structural Comparison ===")
print(phase3)

# ===============================
# Phase 4 — Cross-cohort stress test
# Identify non-Feb cohorts with similar intake conditions
# ===============================
feb_means = feb[numeric_features].mean()

stress_test = []

for month, grp in df.groupby('signup_month'):
    if month == '2024-02':
        continue

    row = {'signup_month': str(month)}
    for feature in numeric_features:
        row[f'{feature}_mean'] = grp[feature].mean()
        row[f'{feature}_diff_from_feb'] = grp[feature].mean() - feb_means[feature]

    row['churn_rate'] = grp[target].mean()
    stress_test.append(row)

phase4 = pd.DataFrame(stress_test)

phase4.to_csv(output_dir / "phase4_cross_cohort_structural_stress_test.csv", index=False)

print("\n=== Phase 4: Cross-Cohort Structural Stress Test ===")
print(phase4.sort_values('churn_rate', ascending=False))

# ===============================
# Inferential exploration completed
# ===============================
print("\nInferential exploration complete.")
print("All outputs saved to:")
print(output_dir)


# ===============================

