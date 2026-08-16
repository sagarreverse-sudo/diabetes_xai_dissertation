from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Define project and data locations
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QOF_DIR = PROJECT_ROOT / "data" / "raw" / "qof_2024_25"


# --------------------------------------------------
# 2. Locate the QOF Excel workbook
# --------------------------------------------------

excel_files = list(QOF_DIR.glob("*.xlsx"))

if len(excel_files) != 1:
    raise RuntimeError(
        f"Expected exactly one Excel file in {QOF_DIR}, "
        f"but found {len(excel_files)}."
    )

qof_file = excel_files[0]

print(f"QOF file: {qof_file.name}")


# --------------------------------------------------
# 3. Load the Diabetes Mellitus (DM) worksheet
# --------------------------------------------------

dm_raw = pd.read_excel(
    qof_file,
    sheet_name="DM",
    header=None
)

print(f"Raw DM worksheet shape: {dm_raw.shape}")


# --------------------------------------------------
# 4. Extract the rows containing ICB observations
# --------------------------------------------------

dm_data = dm_raw.iloc[12:].copy()

valid_icb_mask = (
    dm_data[0].astype("string").str.fullmatch(
        r"Q[A-Z0-9]{2}",
        na=False
    )
    &
    dm_data[1].astype("string").str.fullmatch(
        r"E54[0-9]+",
        na=False
    )
)

dm_icb = dm_data.loc[valid_icb_mask].copy()


# --------------------------------------------------
# 5. Basic structural checks
# --------------------------------------------------

print(f"Valid ICB records: {len(dm_icb)}")
print(f"Unique ODS codes: {dm_icb[0].nunique()}")
print(f"Unique ONS codes: {dm_icb[1].nunique()}")
print(f"Duplicate ODS codes: {dm_icb[0].duplicated().sum()}")
print(f"Duplicate ONS codes: {dm_icb[1].duplicated().sum()}")
