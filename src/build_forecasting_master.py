from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
NDA_FILE = Path("data/processed/nda_predictors_long.csv")
CKD_FILE = Path("data/interim/nda_ckd_type2_icb_2009_2023.csv")

OUTPUT_FILE = Path("data/processed/ckd_forecasting_master.csv")


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
nda = pd.read_csv(NDA_FILE)
ckd = pd.read_csv(CKD_FILE)

print("NDA master shape:", nda.shape)
print("CKD panel shape:", ckd.shape)


# ---------------------------------------------------------
# Standardise ICB codes
# ---------------------------------------------------------
nda["icb_code"] = (
    nda["icb_code"]
    .astype(str)
    .str.strip()
    .str.upper()
)

ckd["icb_code"] = (
    ckd["icb_code"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ---------------------------------------------------------
# Create TARGET CKD table
# ---------------------------------------------------------
target = ckd[
    [
        "icb_code",
        "icb_name",
        "year",
        "ckd_cases",
        "diabetes_population",
        "ckd_risk_rate_per_1000",
    ]
].copy()

target = target.rename(
    columns={
        "year": "target_year",
        "ckd_risk_rate_per_1000": "target_ckd_rate_per_1000",
    }
)


# Keep only target years used in forecasting
target = target[
    target["target_year"].isin([2020, 2021, 2022, 2023])
].copy()


# ---------------------------------------------------------
# Join NDA predictors to future CKD outcome
# ---------------------------------------------------------
master = nda.merge(
    target,
    on=["icb_code", "target_year"],
    how="left",
    validate="one_to_one",
)

print("\nAfter target join:")
print("Shape:", master.shape)


# ---------------------------------------------------------
# Create previous-year CKD predictor
# ---------------------------------------------------------
lag = ckd[
    [
        "icb_code",
        "year",
        "ckd_risk_rate_per_1000",
    ]
].copy()

# Shift year forward so previous-year CKD aligns
# with the following target year.
lag["target_year"] = lag["year"] + 1

lag = lag.rename(
    columns={
        "ckd_risk_rate_per_1000": "previous_year_ckd_rate"
    }
)

lag = lag[
    [
        "icb_code",
        "target_year",
        "previous_year_ckd_rate",
    ]
]


# ---------------------------------------------------------
# Join previous-year CKD
# ---------------------------------------------------------
master = master.merge(
    lag,
    on=["icb_code", "target_year"],
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Validation checks
# ---------------------------------------------------------
expected_rows = 168

if len(master) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} rows, found {len(master)}"
    )

duplicate_rows = master.duplicated(
    subset=["icb_code", "target_year"]
).sum()

missing_target = master[
    "target_ckd_rate_per_1000"
].isna().sum()

missing_lag = master[
    "previous_year_ckd_rate"
].isna().sum()


if duplicate_rows != 0:
    raise ValueError(
        f"Found {duplicate_rows} duplicate ICB-target-year rows"
    )

if missing_target != 0:
    raise ValueError(
        f"Found {missing_target} missing CKD target values"
    )

if missing_lag != 0:
    raise ValueError(
        f"Found {missing_lag} missing previous-year CKD values"
    )
# ---------------------------------------------------------
# Compatibility columns for existing modelling pipeline
# ---------------------------------------------------------
master["audit_year"] = master["predictor_period"]

master["ckd_risk_rate_per_1000"] = (
    master["target_ckd_rate_per_1000"]
)


# ---------------------------------------------------------
# Sort and save
# ---------------------------------------------------------
master = master.sort_values(
    ["target_year", "icb_code"]
).reset_index(drop=True)

master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Audit report
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL FORECASTING MASTER DATASET CREATED")
print("=" * 60)

print(f"Output: {OUTPUT_FILE}")
print(f"Shape: {master.shape}")
print(f"Rows: {len(master)}")
print(f"Unique ICBs: {master['icb_code'].nunique()}")
print(f"Target years: {sorted(master['target_year'].unique())}")

print("\nRows per target year:")
print(master.groupby("target_year").size())

print("\nDuplicate ICB-year rows:", duplicate_rows)
print("Missing target CKD values:", missing_target)
print("Missing previous-year CKD values:", missing_lag)

print("\nCKD alignment check:")
print(
    master[
        [
            "predictor_period",
            "target_year",
            "previous_year_ckd_rate",
            "target_ckd_rate_per_1000",
        ]
    ]
    .groupby(["predictor_period", "target_year"])
    .size()
)

print("\nFINAL MASTER DATASET AUDIT: PASSED")