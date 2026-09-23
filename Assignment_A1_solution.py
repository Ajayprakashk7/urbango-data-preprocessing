# ============================================================
# URBANGO - ASSIGNMENT 1
# DSE ZG505 - Preprocessing for AI-Ready Data
# Student: 2026nd04174@wilp.bits-pilani.ac.in
# ============================================================

import os
import glob
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 0. BASIC SETTINGS
# ============================================================

WORKSPACE = os.path.expanduser("~/workspace")

CLEANED_FILE = os.path.join(
    WORKSPACE,
    "UrbanGo_A1_cleaned.csv"
)

PROFILE_FILE = os.path.join(
    WORKSPACE,
    "profiling_report.html"
)


# ============================================================
# 1. FIND THE ASSIGNED RAW DATASET
# ============================================================

manifest_path = os.path.join(
    WORKSPACE,
    "A1_execution_manifest.json"
)

if not os.path.exists(manifest_path):
    raise FileNotFoundError(
        "ERROR: A1_execution_manifest.json was not found. "
        "Run ./Assignment_A1 first."
    )

csv_files = []

for file in glob.glob(os.path.join(WORKSPACE, "*.csv")):
    filename = os.path.basename(file)

    # Do not accidentally use our final cleaned dataset
    if filename != "UrbanGo_A1_cleaned.csv":
        csv_files.append(file)

if len(csv_files) == 0:
    raise FileNotFoundError(
        "ERROR: No RAW CSV dataset found in ~/workspace."
    )

if len(csv_files) > 1:
    print("WARNING: Multiple CSV files found:")
    for f in csv_files:
        print("   ", os.path.basename(f))

    # Prefer the CSV that matches the institutional email
    email_csv = [
        f for f in csv_files
        if "2026nd04174@wilp.bits-pilani.ac.in" in os.path.basename(f)
    ]

    if len(email_csv) == 1:
        raw_file = email_csv[0]
    else:
        raise RuntimeError(
            "ERROR: More than one possible RAW CSV exists. "
            "Do not guess. Check ~/workspace."
        )
else:
    raw_file = csv_files[0]


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

print("\n" + "=" * 70)
print("URBANGO ASSIGNMENT 1 - PART 1")
print("=" * 70)

print("\nRAW DATASET LOADED SUCCESSFULLY")
print("Rows    :", end=" ")

raw_df = pd.read_csv(raw_file)

print(len(raw_df))
print("Columns :", len(raw_df.columns))

print("\nColumn names:")
for col in raw_df.columns:
    print(" -", col)


# IMPORTANT:
# raw_df is never modified during the assignment.
# All preprocessing is performed on clean_df.


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """
    Find a column using case-insensitive / punctuation-insensitive
    matching.
    """
    normalized = {}

    for col in df.columns:
        key = "".join(
            ch.lower() for ch in str(col)
            if ch.isalnum()
        )
        normalized[key] = col

    for candidate in candidates:
        key = "".join(
            ch.lower() for ch in candidate
            if ch.isalnum()
        )

        if key in normalized:
            return normalized[key]

    return None


# Locate important columns.
AGE_COL = find_column(
    raw_df,
    ["customer_age"]
)

RATING_COL = find_column(
    raw_df,
    ["customer_rating"]
)

FARE_COL = find_column(
    raw_df,
    ["fare_amount"]
)

DISTANCE_COL = find_column(
    raw_df,
    ["distance_km"]
)

RIDES_COL = find_column(
    raw_df,
    ["lifetime_rides"]
)

PAYMENT_COL = find_column(
    raw_df,
    ["payment_type"]
)

CITY_COL = find_column(
    raw_df,
    ["city"]
)

TARGET_COL = find_column(
    raw_df,
    [
        "churn",
        "customer_churn",
        "churn_flag",
        "customer_churn_flag",
        "is_churned",
        "target"
    ]
)

required_columns = {
    "customer_age": AGE_COL,
    "customer_rating": RATING_COL,
    "fare_amount": FARE_COL,
    "distance_km": DISTANCE_COL,
    "lifetime_rides": RIDES_COL,
    "payment_type": PAYMENT_COL,
    "city": CITY_COL
}

print("\nIMPORTANT COLUMNS DETECTED:")
for name, value in required_columns.items():
    print(f"{name:20s}: {value}")

if TARGET_COL is None:
    binary_candidates = []

    for col in raw_df.columns:
        try:
            n_unique = raw_df[col].dropna().nunique()

            if n_unique == 2:
                binary_candidates.append(col)
        except Exception:
            pass

    if len(binary_candidates) == 1:
        TARGET_COL = binary_candidates[0]

print("target column       :", TARGET_COL)

for name, value in required_columns.items():
    if value is None:
        raise ValueError(
            f"ERROR: Required column '{name}' was not found."
        )

if TARGET_COL is None:
    raise ValueError(
        "ERROR: Churn/target column could not be identified. "
        "Check the printed column names."
    )


# ============================================================
# PART 1.1
# DATA QUALITY METRICS
# ============================================================

print("\n" + "=" * 70)
print("Q1.1 - DATA QUALITY METRICS")
print("=" * 70)

# 1. Overall percentage of non-null values
overall_non_null_percentage = (
    raw_df.notna().mean().mean() * 100
)

print(
    f"\n1. Overall percentage of non-null values: "
    f"{overall_non_null_percentage:.2f}%"
)


# 2. Biologically impossible customer ages
impossible_age_rows = (
    (raw_df[AGE_COL] < 18) |
    (raw_df[AGE_COL] > 100)
).sum()

print(
    f"2. Rows with biologically impossible customer_age: "
    f"{impossible_age_rows}"
)


# 3. Distinct casing variations in payment_type
payment_values = (
    raw_df[PAYMENT_COL]
    .dropna()
    .astype(str)
    .str.strip()
)

distinct_payment_variations = payment_values.nunique()

distinct_payment_categories_ignore_case = (
    payment_values.str.lower().nunique()
)

print(
    f"3. Distinct payment_type values/casing variations: "
    f"{distinct_payment_variations}"
)

print(
    f"   Distinct payment categories ignoring case: "
    f"{distinct_payment_categories_ignore_case}"
)


# 4. Exact duplicate rows
duplicate_rows = raw_df.duplicated().sum()

print(
    f"4. Exact duplicate rows: {duplicate_rows}"
)


# ============================================================
# PART 1.2
# SCHEMA METADATA
# ============================================================

print("\n" + "=" * 70)
print("Q1.2 - SCHEMA METADATA")
print("=" * 70)

schema_metadata = {}

for col in raw_df.columns:
    schema_metadata[col] = {
        "data_type": str(raw_df[col].dtype),
        "missing_values": int(raw_df[col].isna().sum())
    }

print(
    json.dumps(
        schema_metadata,
        indent=4
    )
)


# ============================================================
# PART 1.3
# YDATA PROFILING REPORT
# ============================================================

print("\n" + "=" * 70)
print("Q1.3 - YDATA PROFILING")
print("=" * 70)

try:
    from ydata_profiling import ProfileReport

    print("\nGenerating profiling_report.html ...")

    profile = ProfileReport(
        raw_df,
        title="UrbanGo Assignment 1 - Raw Dataset Profiling Report"
    )

    profile.to_file(PROFILE_FILE)

    print("Profiling report created successfully.")
    print("File: profiling_report.html")

except Exception as e:
    print("\nERROR while generating profiling report:")
    print(type(e).__name__, str(e))
    raise


# Actual skewness and kurtosis values
fare_skew = raw_df[FARE_COL].skew()
fare_kurtosis = raw_df[FARE_COL].kurt()

rides_skew = raw_df[RIDES_COL].skew()
rides_kurtosis = raw_df[RIDES_COL].kurt()

print("\nNumerical distribution statistics:")

print(
    f"{FARE_COL} -> "
    f"skewness={fare_skew:.4f}, "
    f"kurtosis={fare_kurtosis:.4f}"
)

print(
    f"{RIDES_COL} -> "
    f"skewness={rides_skew:.4f}, "
    f"kurtosis={rides_kurtosis:.4f}"
)


# TECHNICAL OBSERVATIONS - REQUIRED COMMENTS
#
# Observation 1:
# fare_amount is expected to show positive/right skewness because
# the assignment describes extreme surge-pricing values. A large
# positive kurtosis indicates a heavy tail and extreme observations.
#
# Observation 2:
# lifetime_rides is described as having an exponentially widening
# positive tail. Positive skewness together with high kurtosis
# indicates strong right-tail dispersion and extreme values.
#
# Observation 3:
# customer_age contains missing and/or biologically impossible
# observations, so arithmetic-mean imputation alone is not an
# appropriate final preprocessing strategy.
#
# Observation 4:
# payment_type contains case-sensitive inconsistencies and the raw
# dataset may also contain duplicate rows and incomplete records,
# which are important data-quality issues identified during profiling.


# ============================================================
# PART 1.4
# MEAN IMPUTATION AND VARIANCE
# ============================================================

print("\n" + "=" * 70)
print("Q1.4 - VARIANCE COMPRESSION")
print("=" * 70)

age_variance_before = raw_df[AGE_COL].var()

age_mean = raw_df[AGE_COL].mean()

age_temp = raw_df[AGE_COL].copy()

age_temp = age_temp.fillna(age_mean)

age_variance_after = age_temp.var()

print(
    f"\nVariance of customer_age before mean imputation: "
    f"{age_variance_before:.6f}"
)

print(
    f"Variance of customer_age after mean imputation:  "
    f"{age_variance_after:.6f}"
)

# Mathematical explanation:
# Mean imputation places every missing value exactly at the mean,
# contributing zero squared deviation from the mean. Therefore the
# variance is compressed and distinct demographic clusters can be
# artificially pulled toward the center.


# ============================================================
# PART 2
# CREATE A SEPARATE IN-MEMORY COPY
# ============================================================

print("\n" + "=" * 70)
print("PART 2 - PREPROCESSING")
print("=" * 70)

clean_df = raw_df.copy()

print("\nStarting Part 2 rows:", len(clean_df))


# ============================================================
# Q2.1.1
# CUSTOMER RATING MISSINGNESS INDICATOR
# ============================================================

print("\nQ2.1.1 - CUSTOMER RATING")

rating_missing_before = clean_df[RATING_COL].isna().sum()

missing_indicator_col = "customer_rating_missing"

clean_df[missing_indicator_col] = (
    clean_df[RATING_COL].isna().astype(int)
)

print(
    "Missing customer_rating before imputation:",
    rating_missing_before
)

print(
    "Missingness indicator created:",
    missing_indicator_col
)

# Use the median to fill the rating values after preserving
# the missingness pattern in the indicator column.
rating_median = clean_df[RATING_COL].median()

clean_df[RATING_COL] = (
    clean_df[RATING_COL].fillna(rating_median)
)

print(
    "Missing customer_rating after imputation:",
    clean_df[RATING_COL].isna().sum()
)


# ============================================================
# Q2.1.2
# FARE AMOUNT - MEDIAN IMPUTATION
# ============================================================

print("\nQ2.1.2 - FARE AMOUNT")

fare_missing_before = clean_df[FARE_COL].isna().sum()

fare_median = clean_df[FARE_COL].median()

clean_df[FARE_COL] = (
    clean_df[FARE_COL].fillna(fare_median)
)

print(
    "Missing fare_amount before imputation:",
    fare_missing_before
)

print(
    "Median used:",
    fare_median
)

print(
    "Missing fare_amount after imputation:",
    clean_df[FARE_COL].isna().sum()
)


# ============================================================
# Q2.1.3
# CUSTOMER AGE - SCALED 3-NEAREST-NEIGHBOUR IMPUTATION
# ============================================================

print("\nQ2.1.3 - CUSTOMER AGE KNN IMPUTATION")

age_missing_before = clean_df[AGE_COL].isna().sum()

print(
    "Missing customer_age before KNN:",
    age_missing_before
)

if age_missing_before > 0:

    # Use all numeric features except:
    # - customer_age, because it is the target to be predicted
    # - churn/target, because using the label would cause leakage
    numeric_columns = clean_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    predictor_columns = [
        c for c in numeric_columns
        if c not in [AGE_COL, TARGET_COL]
    ]

    if len(predictor_columns) == 0:
        raise ValueError(
            "ERROR: No numeric predictor features available "
            "for customer_age KNN imputation."
        )

    print(
        "Numeric predictor features used for KNN:"
    )

    for col in predictor_columns:
        print(" -", col)

    # Temporary predictor matrix.
    # Missing predictor values are median-filled ONLY for the
    # distance calculation. This does not alter the final values
    # of those columns.
    X_knn = clean_df[predictor_columns].copy()

    for col in predictor_columns:
        X_knn[col] = X_knn[col].fillna(
            X_knn[col].median()
        )

    # Rows with known age train the KNN regressor.
    known_age_mask = clean_df[AGE_COL].notna()
    missing_age_mask = clean_df[AGE_COL].isna()

    X_known = X_knn.loc[known_age_mask]
    y_known = clean_df.loc[
        known_age_mask,
        AGE_COL
    ].astype(float)

    X_missing = X_knn.loc[missing_age_mask]

    if len(X_known) < 3:
        raise ValueError(
            "ERROR: Fewer than 3 records have known customer_age. "
            "Cannot perform the required 3-neighbour method."
        )

    # Scale predictors before calculating Euclidean distances.
    scaler_X = StandardScaler()

    X_known_scaled = scaler_X.fit_transform(X_known)

    X_missing_scaled = scaler_X.transform(X_missing)

    # Scale the age target as well so that the predicted age can
    # be inverse-transformed afterwards.
    scaler_age = StandardScaler()

    y_known_scaled = scaler_age.fit_transform(
        y_known.to_numpy().reshape(-1, 1)
    ).ravel()

    # Exactly 3 peer records as required by the assignment.
    knn_model = KNeighborsRegressor(
        n_neighbors=3,
        weights="uniform",
        metric="euclidean"
    )

    knn_model.fit(
        X_known_scaled,
        y_known_scaled
    )

    predicted_age_scaled = knn_model.predict(
        X_missing_scaled
    )

    # Inverse-transform the predicted standardized age.
    predicted_age = scaler_age.inverse_transform(
        predicted_age_scaled.reshape(-1, 1)
    ).ravel()

    clean_df.loc[
        missing_age_mask,
        AGE_COL
    ] = predicted_age

print(
    "Missing customer_age after KNN:",
    clean_df[AGE_COL].isna().sum()
)


# ============================================================
# Q2.2.1
# DROP IMPOSSIBLE OPERATIONAL RECORDS
# ============================================================

print("\nQ2.2.1 - REMOVE IMPOSSIBLE RECORDS")

invalid_distance_mask = (
    clean_df[DISTANCE_COL] > 200
)

invalid_fare_mask = (
    clean_df[FARE_COL] < 0
)

invalid_rows_mask = (
    invalid_distance_mask |
    invalid_fare_mask
)

invalid_distance_count = invalid_distance_mask.sum()
invalid_fare_count = invalid_fare_mask.sum()
invalid_total_count = invalid_rows_mask.sum()

print(
    f"Rows with distance_km > 200: "
    f"{invalid_distance_count}"
)

print(
    f"Rows with negative fare_amount: "
    f"{invalid_fare_count}"
)

print(
    f"Rows permanently dropped: "
    f"{invalid_total_count}"
)

clean_df = clean_df.loc[
    ~invalid_rows_mask
].copy()

print(
    "Rows after impossible-record removal:",
    len(clean_df)
)


# ============================================================
# Q2.2.2
# CAP EXTREME FARES USING Q3 + 1.5*IQR
# ============================================================

print("\nQ2.2.2 - FARE OUTLIER CAPPING")

Q1 = clean_df[FARE_COL].quantile(0.25)
Q3 = clean_df[FARE_COL].quantile(0.75)

IQR = Q3 - Q1

upper_boundary = Q3 + (1.5 * IQR)

fare_values_capped = (
    clean_df[FARE_COL] > upper_boundary
).sum()

print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Upper boundary Q3 + 1.5*IQR: {upper_boundary}")
print(
    "Number of fare values capped:",
    fare_values_capped
)

clean_df[FARE_COL] = (
    clean_df[FARE_COL].clip(
        upper=upper_boundary
    )
)


# ============================================================
# Q2.2.3
# LOG TRANSFORM LIFETIME RIDES
# ============================================================

print("\nQ2.2.3 - LIFETIME RIDES TRANSFORMATION")

rides_before_skew = clean_df[RIDES_COL].skew()

print(
    f"lifetime_rides skewness before log1p: "
    f"{rides_before_skew:.6f}"
)

if (clean_df[RIDES_COL] < -1).any():
    raise ValueError(
        "ERROR: lifetime_rides contains values below -1. "
        "The required log1p transformation cannot be applied safely."
    )

# log1p(x) = log(1+x)
# It compresses large positive values while retaining zero values.
clean_df[RIDES_COL] = np.log1p(
    clean_df[RIDES_COL]
)

rides_after_skew = clean_df[RIDES_COL].skew()

print(
    f"lifetime_rides skewness after log1p: "
    f"{rides_after_skew:.6f}"
)


# ============================================================
# Q2.2.4
# CITY CONTEXTUAL BASELINE
# ============================================================

print("\nQ2.2.4 - CITY CONTEXTUAL BASELINE")

city_distance_summary = (
    clean_df
    .groupby(CITY_COL)[DISTANCE_COL]
    .mean()
    .reset_index(name="mean_distance_km")
    .sort_values(
        "mean_distance_km",
        ascending=False
    )
)

print(
    "\nMean distance_km by city:"
)

print(
    city_distance_summary.to_string(
        index=False
    )
)


# ============================================================
# FINAL VALIDATION BEFORE SAVING
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPROCESSED DATASET VALIDATION")
print("=" * 70)

print(
    "\nFinal rows:",
    len(clean_df)
)

print(
    "Final columns:",
    len(clean_df.columns)
)

print(
    "\nRemaining missing values:"
)

remaining_missing = (
    clean_df.isna().sum()
)

remaining_missing = remaining_missing[
    remaining_missing > 0
]

if len(remaining_missing) == 0:
    print("NONE")
else:
    print(remaining_missing)


# Check impossible operational values
remaining_bad_distance = (
    clean_df[DISTANCE_COL] > 200
).sum()

remaining_bad_fare = (
    clean_df[FARE_COL] < 0
).sum()

print(
    "\nRemaining distance_km > 200:",
    remaining_bad_distance
)

print(
    "Remaining negative fare_amount:",
    remaining_bad_fare
)


# ============================================================
# 3.1 SAVE FINAL CLEAN DATASET
# ============================================================

print("\n" + "=" * 70)
print("3.1 - SAVING FINAL CLEAN DATASET")
print("=" * 70)

# IMPORTANT:
# This is the ONLY CSV export performed by this program.
# It happens at the very end of the in-memory preprocessing pipeline.

clean_df.to_csv(
    CLEANED_FILE,
    index=False
)

print(
    "\nUrbanGo_A1_cleaned.csv created successfully."
)

print(
    "Final dataset shape:",
    clean_df.shape
)


# ============================================================
# FINAL CHECK
# ============================================================

if not os.path.exists(CLEANED_FILE):
    raise FileNotFoundError(
        "ERROR: UrbanGo_A1_cleaned.csv was not created."
    )

if os.path.getsize(CLEANED_FILE) == 0:
    raise ValueError(
        "ERROR: UrbanGo_A1_cleaned.csv is empty."
    )

print("\n" + "=" * 70)
print("PART 1 + PART 2 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")
print(" - profiling_report.html")
print(" - UrbanGo_A1_cleaned.csv")

print("\nNext step:")
print("Run:")
print("    ./Assignment_1_part_3")

print("\nDO NOT MODIFY THE RAW CSV.")
print("DO NOT MODIFY A1_execution_manifest.json.")
print("=" * 70)

