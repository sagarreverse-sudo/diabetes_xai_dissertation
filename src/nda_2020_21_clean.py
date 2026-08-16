from pathlib import Path
import re

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NDA_2020_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nda_2020_21"
)

NDA_2021_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nda_2021_22"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Locate source workbooks
# --------------------------------------------------

files_2020 = list(
    NDA_2020_DIR.glob("*.xlsx")
)

files_2021 = list(
    NDA_2021_DIR.glob("*.xlsx")
)

if len(files_2020) != 1:
    raise RuntimeError(
        f"Expected one NDA 2020-21 workbook, "
        f"found {len(files_2020)}."
    )

if len(files_2021) != 1:
    raise RuntimeError(
        f"Expected one NDA 2021-22 workbook, "
        f"found {len(files_2021)}."
    )

file_2020 = files_2020[0]
file_2021 = files_2021[0]

print(
    f"NDA 2020-21 file: {file_2020.name}"
)

print(
    f"NDA 2021-22 mapping file: "
    f"{file_2021.name}"
)


# --------------------------------------------------
# 3. Build CCG -> ICB mapping using GP practice codes
# --------------------------------------------------

ccg_lookup = pd.read_excel(
    file_2020,
    sheet_name="CCG GP Code - name lookup",
    header=1
)

icb_lookup = pd.read_excel(
    file_2021,
    sheet_name="ICB sub-ICB GP practice lookup",
    header=1
)


ccg_lookup = (
    ccg_lookup[
        [
            "GP practice code",
            "CCG code",
        ]
    ]
    .dropna()
    .copy()
)


icb_lookup = (
    icb_lookup[
        [
            "GP practice code",
            "ICB code",
        ]
    ]
    .dropna()
    .copy()
)


for col in ccg_lookup.columns:
    ccg_lookup[col] = (
        ccg_lookup[col]
        .astype("string")
        .str.strip()
    )

for col in icb_lookup.columns:
    icb_lookup[col] = (
        icb_lookup[col]
        .astype("string")
        .str.strip()
    )


practice_bridge = ccg_lookup.merge(
    icb_lookup,
    on="GP practice code",
    how="inner"
)


# --------------------------------------------------
# 4. Validate mapping uniqueness
# --------------------------------------------------

mapping_counts = (
    practice_bridge
    .groupby("CCG code")["ICB code"]
    .nunique()
)

if not (mapping_counts == 1).all():
    problematic = (
        mapping_counts[
            mapping_counts != 1
        ]
    )

    raise RuntimeError(
        "Some CCGs map to more than one ICB:\n"
        f"{problematic}"
    )


ccg_to_icb = (
    practice_bridge[
        [
            "CCG code",
            "ICB code",
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)


print(
    f"CCG -> ICB mappings: "
    f"{len(ccg_to_icb)}"
)

print(
    f"Unique CCGs mapped: "
    f"{ccg_to_icb['CCG code'].nunique()}"
)

print(
    f"Unique ICBs represented: "
    f"{ccg_to_icb['ICB code'].nunique()}"
)


if ccg_to_icb["CCG code"].nunique() != 106:
    raise RuntimeError(
        "Expected 106 historical CCGs."
    )

if ccg_to_icb["ICB code"].nunique() != 42:
    raise RuntimeError(
        "Expected 42 ICBs."
    )


# --------------------------------------------------
# 5. Load 2020-21 Type 2 CP_TT worksheet
# --------------------------------------------------

raw = pd.read_excel(
    file_2020,
    sheet_name="Type 2 and other CP_TT",
    header=None
)

print(
    f"Raw Type 2 CP_TT shape: "
    f"{raw.shape}"
)


# --------------------------------------------------
# 6. Extract CCG summary rows
# --------------------------------------------------

data = raw.iloc[3:].copy()

valid_ccg_codes = set(
    ccg_to_icb["CCG code"]
)

ccg_mask = (
    data[1]
    .astype("string")
    .str.strip()
    .isin(valid_ccg_codes)
    &
    (
        data[1]
        .astype("string")
        .str.strip()
        ==
        data[2]
        .astype("string")
        .str.strip()
    )
)

ccg_data = (
    data.loc[ccg_mask]
    .copy()
)

ccg_data["ccg_code"] = (
    ccg_data[1]
    .astype("string")
    .str.strip()
)


print(
    f"CCG summary rows found: "
    f"{len(ccg_data)}"
)

print(
    f"Unique CCG codes: "
    f"{ccg_data['ccg_code'].nunique()}"
)

print(
    f"Duplicate CCG rows: "
    f"{ccg_data['ccg_code'].duplicated().sum()}"
)


if len(ccg_data) != 106:
    raise RuntimeError(
        f"Expected 106 CCG summary rows, "
        f"found {len(ccg_data)}."
    )

if ccg_data["ccg_code"].duplicated().any():
    raise RuntimeError(
        "Duplicate CCG summary rows detected."
    )


# --------------------------------------------------
# 7. Prepare measure headings
# --------------------------------------------------

measure_labels = raw.iloc[1].ffill()
subheaders = raw.iloc[2]


def clean_column_name(text):
    text = str(text).strip().lower()

    text = text.replace(
        "<=",
        "le"
    )

    text = text.replace(
        "<",
        "lt"
    )

    text = text.replace(
        "%",
        "pct"
    )

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


# --------------------------------------------------
# 8. Attach ICB code to each CCG row
# --------------------------------------------------

ccg_data = ccg_data.merge(
    ccg_to_icb,
    left_on="ccg_code",
    right_on="CCG code",
    how="left",
    validate="one_to_one"
)


if ccg_data["ICB code"].isna().any():
    raise RuntimeError(
        "Some CCG rows could not be mapped "
        "to an ICB."
    )


# --------------------------------------------------
# 9. Aggregate each measure correctly
#
# ICB percentage =
# sum(CCG numerators)
# ------------------- x 100
# sum(CCG denominators)
# --------------------------------------------------

measure_start_columns = [
    col
    for col in raw.columns
    if subheaders[col] == "Numerator"
]


clean = pd.DataFrame({
    "icb_code": sorted(
        ccg_data["ICB code"].unique()
    )
})


for numerator_col in measure_start_columns:

    denominator_col = numerator_col + 1

    measure_name = (
        measure_labels[numerator_col]
    )

    clean_name = clean_column_name(
        measure_name
    )

    temp = ccg_data[
        [
            "ICB code",
            numerator_col,
            denominator_col,
        ]
    ].copy()

    temp = temp.rename(
        columns={
            numerator_col: "numerator",
            denominator_col: "denominator",
        }
    )

    temp["numerator"] = pd.to_numeric(
        temp["numerator"],
        errors="coerce"
    )

    temp["denominator"] = pd.to_numeric(
        temp["denominator"],
        errors="coerce"
    )

    missing_num = (
        temp["numerator"]
        .isna()
        .sum()
    )

    missing_den = (
        temp["denominator"]
        .isna()
        .sum()
    )

    if missing_num != 0 or missing_den != 0:
        raise RuntimeError(
            f"Missing numerator/denominator "
            f"values detected for {clean_name}: "
            f"numerator={missing_num}, "
            f"denominator={missing_den}"
        )

    aggregated = (
        temp
        .groupby(
            "ICB code",
            as_index=False
        )
        .agg(
            numerator=(
                "numerator",
                "sum"
            ),
            denominator=(
                "denominator",
                "sum"
            ),
        )
    )

    if (
        aggregated["denominator"] <= 0
    ).any():
        raise RuntimeError(
            f"Invalid denominator detected "
            f"for {clean_name}."
        )

    aggregated[clean_name] = (
        aggregated["numerator"]
        /
        aggregated["denominator"]
        *
        100
    )

    clean = clean.merge(
        aggregated[
            [
                "ICB code",
                clean_name,
            ]
        ],
        left_on="icb_code",
        right_on="ICB code",
        how="left",
        validate="one_to_one"
    )

    clean = clean.drop(
        columns=["ICB code"]
    )


# --------------------------------------------------
# 10. Add audit year
# --------------------------------------------------

clean.insert(
    0,
    "audit_year",
    "2020_21"
)


# --------------------------------------------------
# 11. Sort consistently
# --------------------------------------------------

clean = (
    clean
    .sort_values("icb_code")
    .reset_index(drop=True)
)


# --------------------------------------------------
# 12. Dataset audit
# --------------------------------------------------

print(
    f"\nClean NDA 2020-21 shape: "
    f"{clean.shape}"
)

print(
    f"Unique ICBs: "
    f"{clean['icb_code'].nunique()}"
)

print(
    f"Duplicate ICBs: "
    f"{clean['icb_code'].duplicated().sum()}"
)


print("\nMissing values:")

print(
    clean
    .isna()
    .sum()
    .to_string()
)


numeric = clean.select_dtypes(
    include="number"
)

summary = pd.DataFrame({
    "min": numeric.min(),
    "max": numeric.max(),
    "mean": numeric.mean().round(2),
    "unique": numeric.nunique(),
})

print("\nVariable summary:")

print(
    summary.to_string()
)


# --------------------------------------------------
# 13. Final validation
# --------------------------------------------------

if len(clean) != 42:
    raise RuntimeError(
        f"Expected 42 ICB rows, "
        f"found {len(clean)}."
    )

if clean["icb_code"].nunique() != 42:
    raise RuntimeError(
        "Expected 42 unique ICB codes."
    )

if clean["icb_code"].duplicated().any():
    raise RuntimeError(
        "Duplicate ICB rows detected."
    )

if clean.isna().any().any():
    raise RuntimeError(
        "Missing values detected "
        "in final ICB dataset."
    )


for col in numeric.columns:

    if (
        (clean[col] < 0).any()
        or
        (clean[col] > 100).any()
    ):
        raise RuntimeError(
            f"Invalid percentage range "
            f"in {col}."
        )


print(
    "\nNDA 2020-21 ICB aggregation "
    "check: PASSED"
)


# --------------------------------------------------
# 14. Save
# --------------------------------------------------

output_file = (
    OUTPUT_DIR
    / "nda_type2_icb_2020_21.csv"
)

clean.to_csv(
    output_file,
    index=False
)

print(
    f"Saved to: {output_file}"
)