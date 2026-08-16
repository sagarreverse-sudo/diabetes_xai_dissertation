from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Historical predictor files
# --------------------------------------------------

FILES = {
    "2019_20": INTERIM_DIR / "nda_type2_icb_2019_20.csv",
    "2020_21": INTERIM_DIR / "nda_type2_icb_2020_21.csv",
    "2021_22": INTERIM_DIR / "nda_type2_icb_2021_22.csv",
}


for year, file_path in FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing predictor file for "
            f"{year}: {file_path}"
        )


# --------------------------------------------------
# 3. Load datasets
# --------------------------------------------------

datasets = {
    year: pd.read_csv(file_path)
    for year, file_path in FILES.items()
}


for year, df in datasets.items():

    print(
        f"{year}: "
        f"{df.shape[0]} rows x "
        f"{df.shape[1]} columns"
    )


# --------------------------------------------------
# 4. Validate ICB structure
# --------------------------------------------------

expected_icbs = None


for year, df in datasets.items():

    if len(df) != 42:
        raise RuntimeError(
            f"{year}: expected 42 rows, "
            f"found {len(df)}."
        )

    if df["icb_code"].nunique() != 42:
        raise RuntimeError(
            f"{year}: expected "
            f"42 unique ICBs."
        )

    if df["icb_code"].duplicated().any():
        raise RuntimeError(
            f"{year}: duplicate "
            f"ICB codes detected."
        )

    current_icbs = set(
        df["icb_code"]
    )

    if expected_icbs is None:
        expected_icbs = current_icbs

    elif current_icbs != expected_icbs:
        raise RuntimeError(
            f"{year}: ICB code set "
            f"does not match other years."
        )


print(
    "\nICB structure across years: PASSED"
)


# --------------------------------------------------
# 5. Identify common predictors
# --------------------------------------------------

ID_COLUMNS = {
    "audit_year",
    "icb_code",
}


feature_sets = [
    set(df.columns) - ID_COLUMNS
    for df in datasets.values()
]


COMMON_FEATURES = sorted(
    set.intersection(
        *feature_sets
    )
)


print(
    f"\nCommon predictors across "
    f"all three years: "
    f"{len(COMMON_FEATURES)}"
)


if len(COMMON_FEATURES) != 21:
    raise RuntimeError(
        f"Expected 21 common predictors, "
        f"found {len(COMMON_FEATURES)}."
    )


# --------------------------------------------------
# 6. Prepare datasets indexed by ICB
# --------------------------------------------------

indexed = {}


for year, df in datasets.items():

    indexed[year] = (
        df
        .set_index("icb_code")
        .sort_index()
    )


# --------------------------------------------------
# 7. Temporal audit
#
# We examine two things:
#
# A) Mean-level changes between audit years
# B) Whether ICB geographic rankings remain stable
#
# Spearman correlation is used for B because it
# compares rank ordering rather than requiring
# identical absolute values.
# --------------------------------------------------

audit_rows = []


for feature in COMMON_FEATURES:

    x19 = indexed["2019_20"][feature]
    x20 = indexed["2020_21"][feature]
    x21 = indexed["2021_22"][feature]


    # ----------------------------------------------
    # Descriptive statistics
    # ----------------------------------------------

    mean_19 = x19.mean()
    mean_20 = x20.mean()
    mean_21 = x21.mean()

    std_19 = x19.std()
    std_20 = x20.std()
    std_21 = x21.std()


    # ----------------------------------------------
    # Mean changes in percentage points
    # ----------------------------------------------

    change_19_to_20 = (
        mean_20 - mean_19
    )

    change_20_to_21 = (
        mean_21 - mean_20
    )

    net_change = (
        mean_21 - mean_19
    )

    max_abs_mean_change = max(
        abs(change_19_to_20),
        abs(change_20_to_21),
    )


    # ----------------------------------------------
    # Geographic rank stability
    # ----------------------------------------------

    rho_19_20 = x19.corr(
        x20,
        method="spearman"
    )

    rho_20_21 = x20.corr(
        x21,
        method="spearman"
    )

    rho_19_21 = x19.corr(
        x21,
        method="spearman"
    )


    min_adjacent_rank_corr = min(
        rho_19_20,
        rho_20_21,
    )


    # ----------------------------------------------
    # Simple audit flags
    #
    # These are diagnostic flags only.
    # They do NOT automatically remove predictors.
    # ----------------------------------------------

    large_year_shift = (
        max_abs_mean_change >= 10
    )

    weak_rank_stability = (
        min_adjacent_rank_corr < 0.50
    )


    audit_rows.append({
        "feature": feature,

        "mean_2019_20": mean_19,
        "mean_2020_21": mean_20,
        "mean_2021_22": mean_21,

        "std_2019_20": std_19,
        "std_2020_21": std_20,
        "std_2021_22": std_21,

        "change_2019_20_to_2020_21":
            change_19_to_20,

        "change_2020_21_to_2021_22":
            change_20_to_21,

        "net_change_2019_20_to_2021_22":
            net_change,

        "max_abs_adjacent_mean_change":
            max_abs_mean_change,

        "spearman_2019_20_vs_2020_21":
            rho_19_20,

        "spearman_2020_21_vs_2021_22":
            rho_20_21,

        "spearman_2019_20_vs_2021_22":
            rho_19_21,

        "min_adjacent_rank_correlation":
            min_adjacent_rank_corr,

        "large_year_shift_flag":
            large_year_shift,

        "weak_rank_stability_flag":
            weak_rank_stability,
    })


# --------------------------------------------------
# 8. Build audit table
# --------------------------------------------------

audit = pd.DataFrame(
    audit_rows
)


numeric_columns = audit.select_dtypes(
    include="number"
).columns


audit[numeric_columns] = (
    audit[numeric_columns]
    .round(3)
)


# Sort by biggest year-to-year shift first.
audit = (
    audit
    .sort_values(
        "max_abs_adjacent_mean_change",
        ascending=False
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# 9. Print mean values by year
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "PREDICTOR MEANS BY AUDIT YEAR"
)

print(
    "========================================"
)


mean_table = audit[
    [
        "feature",
        "mean_2019_20",
        "mean_2020_21",
        "mean_2021_22",
        "max_abs_adjacent_mean_change",
    ]
]


print(
    mean_table.to_string(
        index=False
    )
)


# --------------------------------------------------
# 10. Print largest temporal changes
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "LARGEST YEAR-TO-YEAR SHIFTS"
)

print(
    "========================================"
)


shift_table = audit[
    [
        "feature",
        "change_2019_20_to_2020_21",
        "change_2020_21_to_2021_22",
        "max_abs_adjacent_mean_change",
        "large_year_shift_flag",
    ]
]


print(
    shift_table.to_string(
        index=False
    )
)


# --------------------------------------------------
# 11. Print geographic rank stability
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "GEOGRAPHIC RANK STABILITY"
)

print(
    "========================================"
)


rank_table = audit[
    [
        "feature",
        "spearman_2019_20_vs_2020_21",
        "spearman_2020_21_vs_2021_22",
        "spearman_2019_20_vs_2021_22",
        "weak_rank_stability_flag",
    ]
].sort_values(
    "spearman_2020_21_vs_2021_22"
)


print(
    rank_table.to_string(
        index=False
    )
)


# --------------------------------------------------
# 12. Summary of diagnostic flags
# --------------------------------------------------

large_shift_features = audit.loc[
    audit["large_year_shift_flag"],
    "feature"
].tolist()


weak_stability_features = audit.loc[
    audit["weak_rank_stability_flag"],
    "feature"
].tolist()


print(
    "\n========================================"
)

print(
    "AUDIT FLAGS"
)

print(
    "========================================"
)


print(
    f"\nFeatures with >=10 percentage-point "
    f"adjacent-year mean shift: "
    f"{len(large_shift_features)}"
)

for feature in large_shift_features:
    print(f"- {feature}")


print(
    f"\nFeatures with adjacent-year "
    f"Spearman correlation <0.50: "
    f"{len(weak_stability_features)}"
)

for feature in weak_stability_features:
    print(f"- {feature}")


# --------------------------------------------------
# 13. Save complete audit table
# --------------------------------------------------

output_file = (
    OUTPUT_DIR
    / "temporal_predictor_audit.csv"
)


audit.to_csv(
    output_file,
    index=False
)


print(
    "\nTemporal predictor audit: PASSED"
)

print(
    f"Saved to: {output_file}"
)