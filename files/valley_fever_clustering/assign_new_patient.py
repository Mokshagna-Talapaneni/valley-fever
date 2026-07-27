"""
assign_new_patient.py
-----------------------
Real-world use case: a case worker or clinician has ONE new patient in
front of them and wants to know which existing patient group they most
resemble, so they can apply that group's care-coordination playbook.

This is the piece that turns the project from "a clustering script that
runs once on a static CSV" into something that could plug into an actual
intake workflow: fit once on the historical patient population, persist
the fitted preprocessor + KMeans + GMM, then classify new patients
one at a time as they come in.

Design choices:
  - The preprocessor and KMeans model are refit on the full dataset and
    saved with joblib so this script can run independently of main.py
    (mirrors how a real service would separate "batch retraining" from
    "single-patient scoring").
  - Both a hard KMeans assignment (nearest centroid) and a GMM
    probability distribution over clusters are reported. The GMM
    probabilities let us flag "boundary" patients whose group membership
    is ambiguous (max probability below a threshold) -- these are
    exactly the patients where a human should double check the
    algorithm's suggestion rather than act on it automatically.

Run with:
    python assign_new_patient.py          # runs on a built-in example patient
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from generate_data import generate_patients
from clustering import build_preprocessor, run_kmeans, run_gmm, NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODEL_DIR = "outputs/model"
BOUNDARY_THRESHOLD = 0.65  # if top cluster probability is below this, flag for human review


def _align_gmm_labels_to_kmeans(kmeans_labels: np.ndarray, gmm_labels: np.ndarray, k: int) -> dict:
    """KMeans and GMM each number their clusters arbitrarily (KMeans'
    'cluster 0' has no relationship to GMM's 'cluster 0'), so comparing
    or reporting them side by side without aligning first is misleading
    -- easy bug to miss since both still 'run' without erroring. Build a
    contingency table between the two label sets and use the Hungarian
    algorithm to find the permutation of GMM labels that best matches
    KMeans labels, so 'cluster i' means the same group under both
    algorithms downstream."""
    contingency = np.zeros((k, k), dtype=int)
    for km, gm in zip(kmeans_labels, gmm_labels):
        contingency[km, gm] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)  # maximize overlap
    # col_ind[i] = the gmm label that should be renamed to kmeans label i
    gmm_to_kmeans = {int(gmm_label): int(kmeans_label) for kmeans_label, gmm_label in zip(row_ind, col_ind)}
    return gmm_to_kmeans


def fit_and_save_model(k: int, n_patients: int = 300):
    """Fit preprocessor + KMeans + GMM on the (synthetic) historical
    population and persist them, along with cluster profile summaries,
    for reuse when scoring new patients."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = generate_patients(n_patients=n_patients)
    feature_df = df.drop(columns=["patient_id", "true_archetype"])

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(feature_df)
    if hasattr(X, "toarray"):
        X = X.toarray()

    kmeans_result = run_kmeans(X, k)
    gmm_result, _ = run_gmm(X, k)

    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.joblib")
    joblib.dump(kmeans_result, f"{MODEL_DIR}/kmeans_result.joblib")
    joblib.dump(gmm_result, f"{MODEL_DIR}/gmm_result.joblib")

    # Refit the raw sklearn objects too (ClusteringResult only stores labels/metrics)
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    # covariance_type='tied' -- see clustering.run_gmm() docstring for why
    # (per-component full covariance is unstable/overconfident at this
    # sample size relative to dimensionality after one-hot encoding).
    gmm_model = GaussianMixture(n_components=k, random_state=42, n_init=5, covariance_type="tied", reg_covar=1e-3).fit(X)

    # Align GMM's arbitrary label numbering to KMeans' so "cluster i" means
    # the same group under both algorithms when we report results together.
    gmm_to_kmeans = _align_gmm_labels_to_kmeans(kmeans_model.labels_, gmm_model.predict(X), k)

    joblib.dump(kmeans_model, f"{MODEL_DIR}/kmeans_model.joblib")
    joblib.dump(gmm_model, f"{MODEL_DIR}/gmm_model.joblib")
    joblib.dump(gmm_to_kmeans, f"{MODEL_DIR}/gmm_to_kmeans_label_map.joblib")

    return preprocessor, kmeans_model, gmm_model, feature_df, kmeans_result.labels


def load_model():
    preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.joblib")
    kmeans_model = joblib.load(f"{MODEL_DIR}/kmeans_model.joblib")
    gmm_model = joblib.load(f"{MODEL_DIR}/gmm_model.joblib")
    gmm_to_kmeans = joblib.load(f"{MODEL_DIR}/gmm_to_kmeans_label_map.joblib")
    return preprocessor, kmeans_model, gmm_model, gmm_to_kmeans


def assign_patient(patient: dict, preprocessor, kmeans_model, gmm_model, gmm_to_kmeans: dict) -> dict:
    """Score a single new patient (as a dict of raw feature values)."""
    row = pd.DataFrame([patient])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_new = preprocessor.transform(row)
    if hasattr(X_new, "toarray"):
        X_new = X_new.toarray()

    hard_cluster = int(kmeans_model.predict(X_new)[0])

    raw_gmm_probs = gmm_model.predict_proba(X_new)[0]
    # Re-order GMM's probability vector into KMeans' label numbering using
    # the alignment computed at fit time, so cluster_i means the same
    # group in both "kmeans_cluster" and "gmm_cluster_probabilities" below.
    k = len(raw_gmm_probs)
    aligned_probs = np.zeros(k)
    for gmm_label, kmeans_label in gmm_to_kmeans.items():
        aligned_probs[kmeans_label] = raw_gmm_probs[gmm_label]

    top_prob = float(aligned_probs.max())
    is_boundary_case = top_prob < BOUNDARY_THRESHOLD

    return {
        "kmeans_cluster": hard_cluster,
        "gmm_cluster_probabilities": {f"cluster_{i}": round(float(p), 3) for i, p in enumerate(aligned_probs)},
        "confidence": round(top_prob, 3),
        "boundary_case_flag": is_boundary_case,
        "recommendation": (
            "Ambiguous group membership -- route to a case manager for manual review "
            "rather than auto-applying a group care-pathway."
            if is_boundary_case
            else f"Reasonably confident match to cluster {hard_cluster} -- apply that group's care pathway."
        ),
    }


EXAMPLE_NEW_PATIENT = {
    "fever_severity": 7.5,
    "fatigue_severity": 8.0,
    "cough_severity": 6.5,
    "joint_pain_severity": 6.0,
    "chest_pain_severity": 5.5,
    "rash_present": 0,
    "night_sweats_severity": 6.5,
    "weight_loss_pct": 6.0,
    "symptom_duration_weeks": 10.0,
    "disseminated_disease": 0,
    "housing_stability": 3.5,
    "employment_status": "unemployed",
    "education_level": "hs",
    "social_support_score": 3.0,
    "access_to_care_score": 3.5,
    "insurance_status": "uninsured",
    "distance_to_clinic_mi": 22.0,
    "income_bracket": "<25k",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign a new patient to a Valley Fever risk group.")
    parser.add_argument("--k", type=int, default=2, help="number of clusters to fit (default: best k from main.py run)")
    parser.add_argument("--patient_json", type=str, default=None, help="path to a JSON file with one patient's features")
    args = parser.parse_args()

    print("Fitting model on historical (synthetic) population...")
    preprocessor, kmeans_model, gmm_model, feature_df, labels = fit_and_save_model(k=args.k)
    gmm_to_kmeans = joblib.load(f"{MODEL_DIR}/gmm_to_kmeans_label_map.joblib")
    print(f"Model fit and saved to {MODEL_DIR}/\n")

    if args.patient_json:
        with open(args.patient_json) as f:
            patient = json.load(f)
        result = assign_patient(patient, preprocessor, kmeans_model, gmm_model, gmm_to_kmeans)
        print("Assignment result:")
        print(json.dumps(result, indent=2))
    else:
        print("No --patient_json provided; scoring the built-in example (clear-cut, high-risk) patient:\n")
        print(json.dumps(EXAMPLE_NEW_PATIENT, indent=2))
        result = assign_patient(EXAMPLE_NEW_PATIENT, preprocessor, kmeans_model, gmm_model, gmm_to_kmeans)
        print("\nAssignment result:")
        print(json.dumps(result, indent=2))

        # In this synthetic run the two archetypes are well separated, so
        # boundary cases barely occur naturally on realistic patients (see
        # README) -- and it's not simply a numeric-feature effect: even
        # averaging the two clusters' raw numeric feature means while
        # keeping one cluster's categorical SDOH values (employment,
        # insurance, income) still resolves confidently, suggesting the
        # categorical SDOH features are doing a lot of the separating work
        # (see equity_audit.py / README "equity" section). To prove the
        # boundary-flagging *mechanism* itself works correctly, score the
        # true geometric midpoint between the two cluster centroids
        # directly in the model's transformed feature space, bypassing
        # the raw-feature -> preprocessor step entirely.
        c0, c1 = kmeans_model.cluster_centers_[0], kmeans_model.cluster_centers_[1]
        midpoint_transformed = ((c0 + c1) / 2).reshape(1, -1)
        hard_cluster = int(kmeans_model.predict(midpoint_transformed)[0])
        raw_probs = gmm_model.predict_proba(midpoint_transformed)[0]
        aligned = np.zeros(len(raw_probs))
        for gmm_label, kmeans_label in gmm_to_kmeans.items():
            aligned[kmeans_label] = raw_probs[gmm_label]
        top_prob = float(aligned.max())
        print("\n--- Sanity check: exact midpoint between the two cluster centroids (transformed space) ---")
        print(json.dumps({
            "kmeans_cluster": hard_cluster,
            "gmm_cluster_probabilities": {f"cluster_{i}": round(float(p), 3) for i, p in enumerate(aligned)},
            "confidence": round(top_prob, 3),
            "boundary_case_flag": top_prob < BOUNDARY_THRESHOLD,
        }, indent=2))
