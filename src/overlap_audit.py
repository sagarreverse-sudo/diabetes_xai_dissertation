from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "combined_qof_nda_icb_2024_25.csv"
)


# --------------------------------------------------
# 2. Load combined dataset
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print(f"Combined dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Define conceptually overlapping measures
# --------------------------------------------------

pairs = {
    "Foot surveillance": (
        "dm012_foot_exam_risk_classification",
        "foot_surveillance"
    ),

    "HbA1c <=58": (
        "dm020_hba1c_control_no_frailty",
        "hba1c_le_58_mmol_mol_7_5pct"
    ),

    "Blood pressure <=140/90": (
        "dm033_bp_control",
        "blood_pressure_le_140_90"
    ),

    "Primary prevention statin": (
        "dm022_primary_prevention_statin",
        "primary_prevention_on_statins_without_cvd_history"
    ),

    "Secondary prevention statin": (
        "dm023_secondary_prevention_statin",
        "secondary_prevention_on_statins_with_cvd_history"
    )
}


# --------------------------------------------------
# 4. Compare each pair
# --------------------------------------------------

results = []

for concept, (qof_var, nda_var) in pairs.items():

    pearson = df[qof_var].corr(
        df[nda_var],
        method="pearson"
    )

    spearman = df[qof_var].corr(
        df[nda_var],
        method="spearman"
    )

    mean_abs_difference = (
        df[qof_var] - df[nda_var]
    ).abs().mean()

    results.append({
        "concept": concept,
        "qof_variable": qof_var,
        "nda_variable": nda_var,
        "pearson_r": round(pearson, 3),
        "spearman_rho": round(spearman, 3),
        "mean_abs_difference": round(
            mean_abs_difference,
            2
        )
    })


results_df = pd.DataFrame(results)

print("\nOVERLAPPING VARIABLE AUDIT")
print(results_df.to_string(index=False))