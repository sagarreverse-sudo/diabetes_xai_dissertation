from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Define project and data locations
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NDA_DIR = PROJECT_ROOT / "data" / "raw" / "nda_2024_25"


# --------------------------------------------------
# 2. Locate the NDA workbook
# --------------------------------------------------

excel_files = list(NDA_DIR.glob("*.xlsx"))

if len(excel_files) != 1:
    raise RuntimeError(
        f"Expected exactly one Excel file in {NDA_DIR}, "
        f"but found {len(excel_files)}."
    )

nda_file = excel_files[0]

print(f"NDA file: {nda_file.name}")


# --------------------------------------------------
# 3. Load Type 2 and other care-process data
# --------------------------------------------------

nda_raw = pd.read_excel(
    nda_file,
    sheet_name="Type 2 and other CP_TT",
    header=None
)

print(f"Raw NDA worksheet shape: {nda_raw.shape}")


# --------------------------------------------------
# 4. Extract organisation-level rows
# --------------------------------------------------

nda_data = nda_raw.iloc[9:].copy()

organisation_mask = (
    nda_data[0].astype("string").str.fullmatch(
        r"Q[A-Z0-9]{2}",
        na=False
    )
    &
    nda_data[[2, 3, 4, 5]].isna().all(axis=1)
)

nda_org = nda_data.loc[organisation_mask].copy()

print(f"Organisation-level rows: {len(nda_org)}")


# --------------------------------------------------
# 5. Exclude detained estates
# --------------------------------------------------

nda_icb = nda_org.loc[nda_org[0] != "Q99"].copy()


# --------------------------------------------------
# 6. Structural checks
# --------------------------------------------------

print(f"Statutory ICB records: {len(nda_icb)}")
print(f"Unique ICB codes: {nda_icb[0].nunique()}")
print(f"Duplicate ICB codes: {nda_icb[0].duplicated().sum()}")


# --------------------------------------------------
# 7. Fail if the expected structure is not present
# --------------------------------------------------

if len(nda_icb) != 42:
    raise RuntimeError(
        f"Expected 42 statutory ICBs, but found {len(nda_icb)}."
    )

if nda_icb[0].duplicated().any():
    raise RuntimeError("Duplicate ICB codes detected.")

print("NDA ICB structure check: PASSED")