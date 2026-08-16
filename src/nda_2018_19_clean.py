from pathlib import Path
import re

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NDA_2019_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nda_2019_20"
)

NDA_2021_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nda_2021_22"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
)

INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Locate source workbooks
# --------------------------------------------------

files_2019 = list(
    NDA_2019_DIR.glob("*.xlsx")
)

files_2021 = list(
    NDA_2021_DIR.glob("*.xlsx")
)

if len(files_2019) != 1:
    raise RuntimeError(
        f"Expected one NDA 2019-20 workbook, "
        f"found {len(files_2019)}."
    )

if len(files_2021) != 1:
    raise RuntimeError(
        f"Expected one NDA 2021-22 workbook, "
        f"found {len(files_2021)}."
    )

file_2019 = files_2019[0]
file_2021 = files_2021[0]

print(
    f"NDA historical workbook: "
    f"{file_2019.name}"
)

print(
    f"ICB mapping workbook: "
    f"{file_2021.name}"
)


# --------------------------------------------------
# 3. Load historical Type 2 worksheet
#
# The 2019-20 workbook contains BOTH:
# 2018_19 and 2019_20.
#
# We use only 2018_19 here.
# --------------------------------------------------

raw = pd.read_excel(
    file_2019,
    sheet_name="Type 2 and other CP_TT",
    header=None
)

print(
    f"Raw Type 2 CP_TT shape: "
    f"{raw.shape}"
)

data = raw.iloc[3:].copy()


# --------------------------------------------------
# 4. Standardise hierarchy fields
# --------------------------------------------------

audit_year = (
    data[0]
    .astype("string")
    .str.strip()
)

ccg_field = (
    data[1]
    .astype("string")
    .str.strip()
)

organisation_field = (
    data[2]
    .astype("string")
    .str.strip()
)


# --------------------------------------------------
# 5. Extract 2018-19 CCG summary rows
# --------------------------------------------------

ccg_summary_mask = (
    (audit_year == "2018_19")
    &
    (ccg_field == organisation_field)
    &
    ccg_field.str.fullmatch(
        r"[A-Z0-9]{3}",
        na=False
    )
    &
    (ccg_field != "England")
)

ccg_data = (
    data.loc[ccg_summary_mask]
    .copy()
)

ccg_data["ccg_code"] = (
    ccg_data[1]
    .astype("string")
    .str.strip()
)

print(
    f"2018-19 CCG summary rows: "
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


if len(ccg_data) != 195:
    raise RuntimeError(
        f"Expected 195 CCG summary rows, "
        f"found {len(ccg_data)}."
    )

if ccg_data["ccg_code"].nunique() != 195:
    raise RuntimeError(
        "Expected 195 unique CCG codes."
    )

if ccg_data["ccg_code"].duplicated().any():
    raise RuntimeError(
        "Duplicate 2018-19 CCG rows detected."
    )


# --------------------------------------------------
# 6. Extract 2018-19 GP practice -> CCG mapping
#
# This comes directly from the 2018-19 rows.
# --------------------------------------------------

gp_mask = (
    (audit_year == "2018_19")
    &
    ccg_field.str.fullmatch(
        r"[A-Z0-9]{3}",
        na=False
    )
    &
    organisation_field.str.fullmatch(
        r"[A-Z][0-9]{5}",
        na=False
    )
)

gp_to_ccg = (
    data.loc[
        gp_mask,
        [1, 2]
    ]
    .copy()
)

gp_to_ccg.columns = [
    "CCG code",
    "GP practice code",
]

gp_to_ccg["CCG code"] = (
    gp_to_ccg["CCG code"]
    .astype("string")
    .str.strip()
)

gp_to_ccg["GP practice code"] = (
    gp_to_ccg["GP practice code"]
    .astype("string")
    .str.strip()
)

gp_to_ccg = (
    gp_to_ccg
    .drop_duplicates()
)


print(
    f"2018-19 GP-practice mappings: "
    f"{len(gp_to_ccg)}"
)

print(
    f"CCGs represented by practices: "
    f"{gp_to_ccg['CCG code'].nunique()}"
)


# --------------------------------------------------
# 7. Load 2021-22 GP practice -> ICB lookup
# --------------------------------------------------

icb_lookup = pd.read_excel(
    file_2021,
    sheet_name="ICB sub-ICB GP practice lookup",
    header=1
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

for col in icb_lookup.columns:
    icb_lookup[col] = (
        icb_lookup[col]
        .astype("string")
        .str.strip()
    )


# --------------------------------------------------
# 8. Build 2018-19 CCG -> ICB bridge
#
# 2018-19:
# GP practice -> CCG
#
# 2021-22:
# GP practice -> ICB
#
# Shared GP practice codes create the bridge.
# --------------------------------------------------

practice_bridge = gp_to_ccg.merge(
    icb_lookup,
    on="GP practice code",
    how="inner"
)

print(
    f"Common GP practices with "
    f"2021-22 lookup: "
    f"{practice_bridge['GP practice code'].nunique()}"
)


mapping_counts = (
    practice_bridge
    .groupby("CCG code")["ICB code"]
    .nunique()
)


ambiguous_ccgs = (
    mapping_counts[
        mapping_counts > 1
    ]
)

if len(ambiguous_ccgs) > 0:
    raise RuntimeError(
        "Some 2018-19 CCGs map to "
        "multiple ICBs:\n"
        f"{ambiguous_ccgs}"
    )


mapped_ccgs = set(
    practice_bridge["CCG code"]
)

expected_ccgs = set(
    ccg_data["ccg_code"]
)

unmapped_ccgs = sorted(
    expected_ccgs
    - mapped_ccgs
)


if unmapped_ccgs:
    raise RuntimeError(
        "Unmapped 2018-19 CCGs detected:\n"
        f"{unmapped_ccgs}"
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


if ccg_to_icb["CCG code"].nunique() != 195:
    raise RuntimeError(
        "Expected 195 mapped CCGs."
    )

if ccg_to_icb["ICB code"].nunique() != 42:
    raise RuntimeError(
        "Expected 42 ICBs."
    )


# --------------------------------------------------
# 9. Attach ICB code to CCG summary rows
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
        "Some CCG summary rows could not "
        "be mapped to an ICB."
    )


# --------------------------------------------------
# 10. Prepare measure labels
# --------------------------------------------------

measure_labels = (
    raw.iloc[1]
    .ffill()
)

subheaders = raw.iloc[2]


def clean_column_name(text):
    """
    Convert published NHS measure names
    into consistent Python column names.
    """

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
# 11. Identify published measures
# --------------------------------------------------

numerator_columns = [
    col
    for col in raw.columns
    if subheaders[col] == "Numerator"
]


print(
    f"Published measures detected: "
    f"{len(numerator_columns)}"
)


# --------------------------------------------------
# 12. Measures intentionally excluded
#
# In 2018-19 these contain no numeric data:
#
# - Retinal Screening
# - All Nine Care Processes
#
# We do not impute or invent values.
# --------------------------------------------------

EXCLUDED_FEATURES = {
    "retinal_screening",
    "all_nine_care_processes",
}


print(
    "\nIntentionally excluded "
    "2018-19 measures:"
)

for feature in sorted(
    EXCLUDED_FEATURES
):
    print(f"- {feature}")


# --------------------------------------------------
# 13. Aggregate available CCG measures to ICB
#
# Correct calculation:
#
# sum(CCG numerators)
# ------------------- x 100
# sum(CCG denominators)
#
# We do NOT average CCG percentages.
# --------------------------------------------------

clean = pd.DataFrame({
    "icb_code": sorted(
        ccg_data["ICB code"].unique()
    )
})


included_features = []


for numerator_col in numerator_columns:

    denominator_col = (
        numerator_col + 1
    )

    measure_name = (
        measure_labels[numerator_col]
    )

    clean_name = clean_column_name(
        measure_name
    )


    # Skip measures known to be unavailable
    # in 2018-19.
    if clean_name in EXCLUDED_FEATURES:
        continue


    temp = ccg_data[
        [
            "ICB code",
            numerator_col,
            denominator_col,
        ]
    ].copy()


    temp = temp.rename(
        columns={
            numerator_col:
                "numerator",

            denominator_col:
                "denominator",
        }
    )


    temp["numerator"] = (
        pd.to_numeric(
            temp["numerator"],
            errors="coerce"
        )
    )

    temp["denominator"] = (
        pd.to_numeric(
            temp["denominator"],
            errors="coerce"
        )
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


    if (
        missing_num != 0
        or
        missing_den != 0
    ):
        raise RuntimeError(
            f"Unexpected missing values "
            f"for {clean_name}: "
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
        aggregated["denominator"]
        <= 0
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

    included_features.append(
        clean_name
    )


# --------------------------------------------------
# 14. Add audit year
# --------------------------------------------------

clean.insert(
    0,
    "audit_year",
    "2018_19"
)


# --------------------------------------------------
# 15. Sort consistently
# --------------------------------------------------

clean = (
    clean
    .sort_values("icb_code")
    .reset_index(drop=True)
)


# --------------------------------------------------
# 16. Audit final dataset
# --------------------------------------------------

print(
    f"\nIncluded available measures: "
    f"{len(included_features)}"
)

print(
    f"Clean NDA 2018-19 shape: "
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


print(
    "\nMissing values:"
)

print(
    clean
    .isna()
    .sum()
    .to_string()
)


# --------------------------------------------------
# 17. Variable summary
# --------------------------------------------------

numeric = (
    clean
    .select_dtypes(
        include="number"
    )
)


summary = pd.DataFrame({
    "min":
        numeric.min(),

    "max":
        numeric.max(),

    "mean":
        numeric.mean().round(2),

    "unique":
        numeric.nunique(),
})


print(
    "\nVariable summary:"
)

print(
    summary.to_string()
)


# --------------------------------------------------
# 18. Validate against CKD 2020 geography
# --------------------------------------------------

CKD_FILE = (
    INTERIM_DIR
    / "nda_ckd_type2_icb_2009_2023.csv"
)


if not CKD_FILE.exists():
    raise FileNotFoundError(
        f"CKD outcome file not found: "
        f"{CKD_FILE}"
    )


ckd = pd.read_csv(
    CKD_FILE
)


ckd_2020_codes = set(
    ckd.loc[
        ckd["year"] == 2020,
        "icb_code"
    ]
)


predictor_codes = set(
    clean["icb_code"]
)


print(
    "\nCKD 2020 geography check:"
)

print(
    f"2018-19 predictor ICBs: "
    f"{len(predictor_codes)}"
)

print(
    f"CKD 2020 ICBs: "
    f"{len(ckd_2020_codes)}"
)

print(
    f"Common ICBs: "
    f"{len(predictor_codes & ckd_2020_codes)}"
)


only_predictors = sorted(
    predictor_codes
    - ckd_2020_codes
)

only_ckd = sorted(
    ckd_2020_codes
    - predictor_codes
)


print(
    f"Only predictors: "
    f"{only_predictors}"
)

print(
    f"Only CKD: "
    f"{only_ckd}"
)


if predictor_codes != ckd_2020_codes:
    raise RuntimeError(
        "2018-19 predictor geography "
        "does not match CKD 2020."
    )


# --------------------------------------------------
# 19. Final validation rules
# --------------------------------------------------

if len(clean) != 42:
    raise RuntimeError(
        f"Expected 42 final ICB rows, "
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
        "in final 2018-19 dataset."
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


if (
    "retinal_screening"
    in clean.columns
):
    raise RuntimeError(
        "Retinal screening should not "
        "be present in 2018-19 output."
    )


if (
    "all_nine_care_processes"
    in clean.columns
):
    raise RuntimeError(
        "All-nine care processes should not "
        "be present in 2018-19 output."
    )


print(
    "\nNDA 2018-19 ICB aggregation "
    "check: PASSED"
)


# --------------------------------------------------
# 20. Save cleaned dataset
# --------------------------------------------------

OUTPUT_FILE = (
    INTERIM_DIR
    / "nda_type2_icb_2018_19.csv"
)


clean.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"Saved to: {OUTPUT_FILE}"
)