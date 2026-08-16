from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Define project locations
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QOF_DIR = PROJECT_ROOT / "data" / "raw" / "qof_2024_25"
NDA_DIR = PROJECT_ROOT / "data" / "raw" / "nda_2024_25"


# --------------------------------------------------
# 2. Locate source files
# --------------------------------------------------

qof_files = list(QOF_DIR.glob("*.xlsx"))
nda_files = list(NDA_DIR.glob("*.xlsx"))

if len(qof_files) != 1:
    raise RuntimeError(
        f"Expected one QOF workbook, found {len(qof_files)}."
    )

if len(nda_files) != 1:
    raise RuntimeError(
        f"Expected one NDA workbook, found {len(nda_files)}."
    )

qof_file = qof_files[0]
nda_file = nda_files[0]


# --------------------------------------------------
# 3. Extract QOF statutory ICB codes
# --------------------------------------------------

qof_raw = pd.read_excel(
    qof_file,
    sheet_name="DM",
    header=None
)

qof_data = qof_raw.iloc[12:].copy()

qof_mask = (
    qof_data[0]
    .astype("string")
    .str.fullmatch(r"Q[A-Z0-9]{2}", na=False)
)

qof_codes = set(qof_data.loc[qof_mask, 0])


# --------------------------------------------------
# 4. Extract NDA statutory ICB codes
# --------------------------------------------------

nda_raw = pd.read_excel(
    nda_file,
    sheet_name="Type 2 and other CP_TT",
    header=None
)

nda_data = nda_raw.iloc[9:].copy()

nda_mask = (
    nda_data[0]
    .astype("string")
    .str.fullmatch(r"Q[A-Z0-9]{2}", na=False)
    &
    nda_data[[2, 3, 4, 5]].isna().all(axis=1)
    &
    (nda_data[0] != "Q99")
)

nda_codes = set(nda_data.loc[nda_mask, 0])


# --------------------------------------------------
# 5. Compare ICB code sets
# --------------------------------------------------

common_codes = qof_codes & nda_codes
only_qof = qof_codes - nda_codes
only_nda = nda_codes - qof_codes

print(f"QOF statutory ICBs: {len(qof_codes)}")
print(f"NDA statutory ICBs: {len(nda_codes)}")
print(f"Common ICBs: {len(common_codes)}")
print(f"Only in QOF: {sorted(only_qof)}")
print(f"Only in NDA: {sorted(only_nda)}")


# --------------------------------------------------
# 6. Validate linkage
# --------------------------------------------------

if qof_codes != nda_codes:
    raise RuntimeError(
        "QOF and NDA ICB code sets do not match."
    )

if len(common_codes) != 42:
    raise RuntimeError(
        f"Expected 42 common statutory ICBs, "
        f"but found {len(common_codes)}."
    )

print("QOF-NDA ICB linkage check: PASSED")