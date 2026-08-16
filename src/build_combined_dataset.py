from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


# --------------------------------------------------
# 2. Load cleaned QOF and NDA datasets
# --------------------------------------------------

qof_file = INTERIM_DIR / "qof_diabetes_icb_2024_25.csv"
nda_file = INTERIM_DIR / "nda_type2_icb_2024_25.csv"

qof = pd.read_csv(qof_file)
nda = pd.read_csv(nda_file)


# --------------------------------------------------
# 3. Check source structures
# --------------------------------------------------

print(f"QOF shape: {qof.shape}")
print(f"NDA shape: {nda.shape}")

print(f"QOF unique ICBs: {qof['icb_code'].nunique()}")
print(f"NDA unique ICBs: {nda['icb_code'].nunique()}")


# --------------------------------------------------
# 4. Merge on ICB code
# --------------------------------------------------

combined = qof.merge(
    nda,
    on="icb_code",
    how="inner",
    suffixes=("_qof", "_nda"),
    validate="one_to_one"
)


# --------------------------------------------------
# 5. Validate merge
# --------------------------------------------------

print(f"\nCombined shape: {combined.shape}")
print(f"Combined unique ICBs: {combined['icb_code'].nunique()}")

if len(combined) != 42:
    raise RuntimeError(
        f"Expected 42 merged ICBs, found {len(combined)}."
    )


# --------------------------------------------------
# 6. Check whether organisation names agree
# --------------------------------------------------

# --------------------------------------------------
# 6. Check whether organisation names agree
#    after removing formatting differences
# --------------------------------------------------

def normalise_icb_name(series):
    return (
        series
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(
            "integrated care board",
            "icb",
            regex=False
        )
        .str.replace(r"^nhs\s+", "", regex=True)
        .str.replace(r"\s+icb$", "", regex=True)
        .str.strip()
    )


qof_names = normalise_icb_name(
    combined["icb_name_qof"]
)

nda_names = normalise_icb_name(
    combined["icb_name_nda"]
)

name_match = qof_names == nda_names

print(
    f"Normalised ICB name matches: "
    f"{name_match.sum()} / 42"
)

if not name_match.all():
    print("\nUnexpected name differences:")
    print(
        combined.loc[
            ~name_match,
            [
                "icb_code",
                "icb_name_qof",
                "icb_name_nda"
            ]
        ].to_string(index=False)
    )


# Use the fuller QOF organisation name
# as the canonical name in the combined table
combined["icb_name"] = combined["icb_name_qof"]

combined = combined.drop(
    columns=[
        "icb_name_qof",
        "icb_name_nda"
    ]
)


# --------------------------------------------------
# 7. Save combined interim dataset
# --------------------------------------------------

output_file = INTERIM_DIR / "combined_qof_nda_icb_2024_25.csv"

combined.to_csv(
    output_file,
    index=False
)

print(f"\nSaved to: {output_file}")