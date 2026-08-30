from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ckd_forecasting_master.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "feature_set_manifest.csv"
)


# --------------------------------------------------
# 2. Full 19-feature set
#
# These are the predictors consistently available
# across all four historical NDA periods.
# --------------------------------------------------

FULL_FEATURES = [
    "hba1c",
    "blood_pressure",
    "cholesterol",
    "serum_creatinine",
    "urine_albumin",
    "foot_surveillance",
    "bmi",
    "smoking",
    "all_eight_care_processes",

    "hba1c_le_48_mmol_mol_6_5pct",
    "hba1c_le_53_mmol_mol_7_0pct",
    "hba1c_le_58_mmol_mol_7_5pct",
    "hba1c_le_75_mmol_mol_9_0pct",
    "hba1c_le_86_mmol_mol_10_0pct",

    "blood_pressure_le_140_80",

    "primary_prevention_on_statins_without_cvd_history",
    "secondary_prevention_on_statins_with_cvd_history",
    "combined_prevention_on_statins",

    "all_three_treatment_targets",
]


# --------------------------------------------------
# 3. Pre-specified compact primary feature set
#
# IMPORTANT:
# These features are selected before predictive
# model-performance testing.
#
# Selection is based on:
# - clinical relevance to diabetic CKD risk
# - temporal stability evidence
# - redundancy audit
# - parsimony / interpretability
# --------------------------------------------------

PRIMARY_FEATURES = [
    "urine_albumin",
    "serum_creatinine",
    "hba1c_le_58_mmol_mol_7_5pct",
    "blood_pressure_le_140_80",
    "combined_prevention_on_statins",
]


# --------------------------------------------------
# 4. Feature metadata
# --------------------------------------------------

FEATURE_METADATA = {

    "hba1c": {
        "domain": "Care process",
        "reason":
            "HbA1c measurement completion; retained "
            "in full sensitivity set but not primary "
            "set because target-achievement HbA1c "
            "measure is more directly interpretable."
    },

    "blood_pressure": {
        "domain": "Care process",
        "reason":
            "Blood-pressure measurement completion; "
            "retained for sensitivity analysis."
    },

    "cholesterol": {
        "domain": "Care process",
        "reason":
            "Cholesterol measurement completion; "
            "temporally unstable on both mean-shift "
            "and rank-stability evidence, therefore "
            "not selected for compact primary set."
    },

    "serum_creatinine": {
        "domain": "Renal monitoring",
        "reason":
            "Directly relevant renal-monitoring "
            "care process and retained in compact "
            "primary model."
    },

    "urine_albumin": {
        "domain": "Renal monitoring",
        "reason":
            "Directly relevant renal-monitoring "
            "care process and retained in compact "
            "primary model."
    },

    "foot_surveillance": {
        "domain": "Care process",
        "reason":
            "Relevant diabetes care process but "
            "not directly kidney-specific; retained "
            "for full sensitivity analysis."
    },

    "bmi": {
        "domain": "Care process",
        "reason":
            "BMI recording completion; retained in "
            "full sensitivity analysis rather than "
            "compact primary model."
    },

    "smoking": {
        "domain": "Care process",
        "reason":
            "Smoking-status recording completion; "
            "retained in full sensitivity analysis."
    },

    "all_eight_care_processes": {
        "domain": "Composite care process",
        "reason":
            "Broad composite measure with substantial "
            "temporal movement and overlap with "
            "individual care-process measures; "
            "not selected for compact primary set."
    },

    "hba1c_le_48_mmol_mol_6_5pct": {
        "domain": "Glycaemic control",
        "reason":
            "Highly correlated with neighbouring "
            "HbA1c thresholds; retained only in "
            "full sensitivity set."
    },

    "hba1c_le_53_mmol_mol_7_0pct": {
        "domain": "Glycaemic control",
        "reason":
            "Highly correlated with neighbouring "
            "HbA1c thresholds; retained only in "
            "full sensitivity set."
    },

    "hba1c_le_58_mmol_mol_7_5pct": {
        "domain": "Glycaemic control",
        "reason":
            "Selected as one representative HbA1c "
            "control threshold to avoid including "
            "multiple strongly redundant thresholds."
    },

    "hba1c_le_75_mmol_mol_9_0pct": {
        "domain": "Glycaemic control",
        "reason":
            "Highly correlated with neighbouring "
            "HbA1c thresholds; retained only in "
            "full sensitivity set."
    },

    "hba1c_le_86_mmol_mol_10_0pct": {
        "domain": "Glycaemic control",
        "reason":
            "Highly correlated with neighbouring "
            "HbA1c thresholds; retained only in "
            "full sensitivity set."
    },

    "blood_pressure_le_140_80": {
        "domain": "Blood-pressure control",
        "reason":
            "Stable treatment-target measure with "
            "clear clinical interpretation; retained "
            "in compact primary model."
    },

    "primary_prevention_on_statins_without_cvd_history": {
        "domain": "Lipid management",
        "reason":
            "Extremely highly correlated with "
            "combined statin measure; excluded from "
            "compact model to reduce redundancy."
    },

    "secondary_prevention_on_statins_with_cvd_history": {
        "domain": "Lipid management",
        "reason":
            "Very temporally stable but represents "
            "a narrower subgroup; retained in full "
            "sensitivity analysis."
    },

    "combined_prevention_on_statins": {
        "domain": "Lipid management",
        "reason":
            "Selected as broader representative "
            "statin-management measure while avoiding "
            "strong redundancy with primary-prevention "
            "statin measure."
    },

    "all_three_treatment_targets": {
        "domain": "Composite treatment control",
        "reason":
            "Stable composite treatment-target "
            "measure but overlaps conceptually with "
            "individual HbA1c and blood-pressure "
            "targets; retained for sensitivity analysis."
    },
}


# --------------------------------------------------
# 5. Load panel and validate available predictors
# --------------------------------------------------

if not PANEL_FILE.exists():
    raise FileNotFoundError(
        f"Forecasting panel not found: {PANEL_FILE}"
    )


panel = pd.read_csv(PANEL_FILE)


print(
    f"Forecasting panel shape: {panel.shape}"
)


if len(panel) != 168:
    raise RuntimeError(
        f"Expected 168 rows, found {len(panel)}."
    )


# --------------------------------------------------
# 6. Validate full feature set
# --------------------------------------------------

if len(FULL_FEATURES) != 19:
    raise RuntimeError(
        f"Expected 19 full predictors, "
        f"found {len(FULL_FEATURES)}."
    )


if len(set(FULL_FEATURES)) != 19:
    raise RuntimeError(
        "Duplicate predictors detected "
        "in FULL_FEATURES."
    )


missing_full = (
    set(FULL_FEATURES)
    - set(panel.columns)
)


if missing_full:
    raise RuntimeError(
        "Full feature set contains columns "
        "missing from forecasting panel: "
        f"{sorted(missing_full)}"
    )


# --------------------------------------------------
# 7. Validate primary feature set
# --------------------------------------------------

if len(PRIMARY_FEATURES) != 5:
    raise RuntimeError(
        f"Expected 5 primary predictors, "
        f"found {len(PRIMARY_FEATURES)}."
    )


if len(set(PRIMARY_FEATURES)) != 5:
    raise RuntimeError(
        "Duplicate predictors detected "
        "in PRIMARY_FEATURES."
    )


if not set(PRIMARY_FEATURES).issubset(
    set(FULL_FEATURES)
):
    raise RuntimeError(
        "Primary feature set must be "
        "a subset of full feature set."
    )


# --------------------------------------------------
# 8. Validate metadata
# --------------------------------------------------

metadata_features = set(
    FEATURE_METADATA.keys()
)


if metadata_features != set(FULL_FEATURES):

    missing_metadata = (
        set(FULL_FEATURES)
        - metadata_features
    )

    extra_metadata = (
        metadata_features
        - set(FULL_FEATURES)
    )

    raise RuntimeError(
        "Feature metadata mismatch.\n"
        f"Missing metadata: "
        f"{sorted(missing_metadata)}\n"
        f"Extra metadata: "
        f"{sorted(extra_metadata)}"
    )


# --------------------------------------------------
# 9. Build reproducible feature-set manifest
# --------------------------------------------------

rows = []


for feature in FULL_FEATURES:

    metadata = FEATURE_METADATA[
        feature
    ]

    rows.append({
        "feature": feature,

        "domain":
            metadata["domain"],

        "primary_compact_set":
            feature in PRIMARY_FEATURES,

        "full_sensitivity_set":
            True,

        "selection_reason":
            metadata["reason"],
    })


manifest = pd.DataFrame(
    rows
)


# --------------------------------------------------
# 10. Add primary-set order
#
# Makes the exact intended model-input ordering
# explicit and reproducible.
# --------------------------------------------------

primary_order = {
    feature: index + 1
    for index, feature
    in enumerate(
        PRIMARY_FEATURES
    )
}


manifest[
    "primary_order"
] = (
    manifest["feature"]
    .map(primary_order)
)


# --------------------------------------------------
# 11. Print feature-set definitions
# --------------------------------------------------

print(
    "\n========================================"
)

print(
    "PRE-SPECIFIED FEATURE SETS"
)

print(
    "========================================"
)


print(
    "\nPRIMARY COMPACT SET"
)

print(
    f"Predictors: {len(PRIMARY_FEATURES)}"
)


for number, feature in enumerate(
    PRIMARY_FEATURES,
    start=1
):

    print(
        f"{number}. {feature}"
    )


print(
    "\nFULL SENSITIVITY SET"
)

print(
    f"Predictors: {len(FULL_FEATURES)}"
)


for number, feature in enumerate(
    FULL_FEATURES,
    start=1
):

    print(
        f"{number}. {feature}"
    )


# --------------------------------------------------
# 12. Primary domain check
# --------------------------------------------------

primary_manifest = (
    manifest[
        manifest[
            "primary_compact_set"
        ]
    ]
    .copy()
)


print(
    "\nPrimary compact model domains:"
)


print(
    primary_manifest[
        [
            "feature",
            "domain",
        ]
    ]
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 13. Final validation
# --------------------------------------------------

if len(manifest) != 19:
    raise RuntimeError(
        f"Expected 19 manifest rows, "
        f"found {len(manifest)}."
    )


if (
    manifest[
        "primary_compact_set"
    ]
    .sum()
    != 5
):
    raise RuntimeError(
        "Expected exactly 5 primary "
        "compact predictors."
    )


if not (
    manifest[
        "full_sensitivity_set"
    ]
).all():
    raise RuntimeError(
        "All 19 predictors must belong "
        "to full sensitivity set."
    )


if manifest[
    "selection_reason"
].str.strip().eq("").any():
    raise RuntimeError(
        "Empty feature-selection "
        "reason detected."
    )


print(
    "\nFeature-set manifest validation: PASSED"
)


# --------------------------------------------------
# 14. Important methodological statement
# --------------------------------------------------

print(
    """
Methodological rule:
The compact primary feature set is defined
before predictive model-performance testing.

The 2023 final test outcome must not be used
to change these feature definitions.

The full 19-feature set will be retained as
a sensitivity comparison rather than treated
as an automatic replacement for the compact
model.
""".strip()
)


# --------------------------------------------------
# 15. Save manifest
# --------------------------------------------------

manifest.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved to: {OUTPUT_FILE}"
)