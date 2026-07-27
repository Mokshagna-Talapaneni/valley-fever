"""
generate_data.py
-----------------
Creates a synthetic dataset of Valley Fever (coccidioidomycosis) patients
with clinical symptom features and social determinants of health (SDOH)
features.

DESIGN NOTE ON SYNTHETIC DATA
==============================
No real patient data is used anywhere in this project (none was provided,
and using real identifiable health data would raise privacy/IRB concerns
that are out of scope for a take-home exercise). Instead, this script
builds a *structured* synthetic population with three latent archetypes
baked in on purpose:

  1. "Mild / well-resourced"   - mild symptoms, stable housing, good
                                  access to care, strong social support.
  2. "Moderate / under-resourced" - moderate-to-severe symptoms, some
                                  housing/employment instability, patchy
                                  access to care.
  3. "Severe / high social risk"  - severe/disseminated symptoms, high
                                  housing instability, unemployed, low
                                  social support, poor access to care.

These archetypes mirror a real epidemiological pattern reported for
Valley Fever in Arizona/California: outdoor/manual laborers and people
with unstable housing or limited healthcare access tend to present later
and with more severe disease. Baking in known clusters lets us sanity
check that the clustering pipeline in clustering.py actually recovers a
sensible structure, rather than evaluating it blind.

Each archetype is a multivariate Gaussian-ish mixture over the feature
set below, with noise added so clusters overlap somewhat (as they would
in real data) rather than being trivially separable.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_patients(n_patients: int = 300, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic Valley Fever patient dataset.

    Features
    --------
    Clinical / symptom features (0-10 severity scales unless noted):
        fever_severity
        fatigue_severity
        cough_severity
        joint_pain_severity
        chest_pain_severity
        rash_present            (0/1)
        night_sweats_severity
        weight_loss_pct         (% body weight lost, continuous)
        symptom_duration_weeks  (continuous, weeks since onset)
        disseminated_disease    (0/1, spread beyond lungs - severe marker)

    Social determinants of health (SDOH) features:
        housing_stability       (0-10, 10 = very stable)
        employment_status       (categorical: employed_ft, employed_pt,
                                  unemployed, disabled, retired)
        education_level         (categorical: <hs, hs, some_college,
                                  bachelors_plus)
        social_support_score    (0-10, 10 = strong support network)
        access_to_care_score    (0-10, 10 = excellent access)
        insurance_status        (categorical: private, medicaid, medicare,
                                  uninsured)
        distance_to_clinic_mi   (continuous, miles to nearest clinic)
        income_bracket          (categorical: <25k, 25-50k, 50-75k, 75k+)

    Returns
    -------
    pd.DataFrame with one row per synthetic patient plus a hidden
    'true_archetype' column (kept only for internal validation plots;
    dropped before being handed to the clustering algorithm).
    """
    rng = np.random.default_rng(seed)

    archetypes = ["mild_well_resourced", "moderate_under_resourced", "severe_high_risk"]
    # Roughly realistic mixture proportions (most VF cases are mild/self-limited)
    weights = [0.5, 0.32, 0.18]
    assignments = rng.choice(archetypes, size=n_patients, p=weights)

    rows = []
    for arch in assignments:
        if arch == "mild_well_resourced":
            fever = rng.normal(3, 1.3)
            fatigue = rng.normal(3.5, 1.5)
            cough = rng.normal(3, 1.4)
            joint = rng.normal(2, 1.3)
            chest = rng.normal(1.5, 1.1)
            rash = rng.binomial(1, 0.10)
            sweats = rng.normal(2, 1.2)
            wloss = rng.normal(1.5, 1.0)
            duration = rng.normal(3, 1.5)
            disseminated = rng.binomial(1, 0.01)

            housing = rng.normal(8.2, 1.2)
            employment = rng.choice(
                ["employed_ft", "employed_pt", "unemployed", "disabled", "retired"],
                p=[0.65, 0.15, 0.05, 0.03, 0.12],
            )
            education = rng.choice(
                ["<hs", "hs", "some_college", "bachelors_plus"], p=[0.05, 0.20, 0.30, 0.45]
            )
            social_support = rng.normal(7.8, 1.3)
            access_care = rng.normal(8.0, 1.2)
            insurance = rng.choice(
                ["private", "medicaid", "medicare", "uninsured"], p=[0.70, 0.10, 0.15, 0.05]
            )
            distance = rng.normal(6, 3)
            income = rng.choice(["<25k", "25-50k", "50-75k", "75k+"], p=[0.05, 0.20, 0.30, 0.45])

        elif arch == "moderate_under_resourced":
            fever = rng.normal(5.5, 1.4)
            fatigue = rng.normal(6, 1.5)
            cough = rng.normal(5.5, 1.4)
            joint = rng.normal(5, 1.6)
            chest = rng.normal(4, 1.5)
            rash = rng.binomial(1, 0.25)
            sweats = rng.normal(5, 1.5)
            wloss = rng.normal(4, 1.8)
            duration = rng.normal(7, 2.5)
            disseminated = rng.binomial(1, 0.05)

            housing = rng.normal(5.0, 1.6)
            employment = rng.choice(
                ["employed_ft", "employed_pt", "unemployed", "disabled", "retired"],
                p=[0.30, 0.30, 0.20, 0.10, 0.10],
            )
            education = rng.choice(
                ["<hs", "hs", "some_college", "bachelors_plus"], p=[0.20, 0.35, 0.30, 0.15]
            )
            social_support = rng.normal(5.0, 1.6)
            access_care = rng.normal(4.8, 1.6)
            insurance = rng.choice(
                ["private", "medicaid", "medicare", "uninsured"], p=[0.30, 0.35, 0.15, 0.20]
            )
            distance = rng.normal(15, 6)
            income = rng.choice(["<25k", "25-50k", "50-75k", "75k+"], p=[0.30, 0.40, 0.20, 0.10])

        else:  # severe_high_risk
            fever = rng.normal(8, 1.2)
            fatigue = rng.normal(8.5, 1.1)
            cough = rng.normal(7.5, 1.3)
            joint = rng.normal(7, 1.5)
            chest = rng.normal(7, 1.5)
            rash = rng.binomial(1, 0.40)
            sweats = rng.normal(7.5, 1.3)
            wloss = rng.normal(8, 2.0)
            duration = rng.normal(13, 4)
            disseminated = rng.binomial(1, 0.35)

            housing = rng.normal(2.5, 1.5)
            employment = rng.choice(
                ["employed_ft", "employed_pt", "unemployed", "disabled", "retired"],
                p=[0.08, 0.12, 0.45, 0.25, 0.10],
            )
            education = rng.choice(
                ["<hs", "hs", "some_college", "bachelors_plus"], p=[0.40, 0.35, 0.20, 0.05]
            )
            social_support = rng.normal(2.8, 1.5)
            access_care = rng.normal(2.5, 1.4)
            insurance = rng.choice(
                ["private", "medicaid", "medicare", "uninsured"], p=[0.08, 0.35, 0.12, 0.45]
            )
            distance = rng.normal(28, 9)
            income = rng.choice(["<25k", "25-50k", "50-75k", "75k+"], p=[0.55, 0.30, 0.10, 0.05])

        rows.append(
            dict(
                fever_severity=fever,
                fatigue_severity=fatigue,
                cough_severity=cough,
                joint_pain_severity=joint,
                chest_pain_severity=chest,
                rash_present=rash,
                night_sweats_severity=sweats,
                weight_loss_pct=wloss,
                symptom_duration_weeks=duration,
                disseminated_disease=disseminated,
                housing_stability=housing,
                employment_status=employment,
                education_level=education,
                social_support_score=social_support,
                access_to_care_score=access_care,
                insurance_status=insurance,
                distance_to_clinic_mi=distance,
                income_bracket=income,
                true_archetype=arch,  # kept for internal validation only
            )
        )

    df = pd.DataFrame(rows)

    # Clip continuous scales to plausible ranges and round for readability
    severity_cols = [
        "fever_severity",
        "fatigue_severity",
        "cough_severity",
        "joint_pain_severity",
        "chest_pain_severity",
        "night_sweats_severity",
        "housing_stability",
        "social_support_score",
        "access_to_care_score",
    ]
    for c in severity_cols:
        df[c] = _clip(df[c], 0, 10).round(1)

    df["weight_loss_pct"] = _clip(df["weight_loss_pct"], 0, 25).round(1)
    df["symptom_duration_weeks"] = _clip(df["symptom_duration_weeks"], 0.5, 40).round(1)
    df["distance_to_clinic_mi"] = _clip(df["distance_to_clinic_mi"], 0.5, 80).round(1)

    df.insert(0, "patient_id", [f"P{str(i+1).zfill(4)}" for i in range(n_patients)])

    return df


if __name__ == "__main__":
    df = generate_patients()
    out_path = "outputs/synthetic_patients.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic patients -> {out_path}")
    print(df.drop(columns=["true_archetype"]).head())
