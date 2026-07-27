"""
test_pipeline.py
-----------------
Lightweight sanity tests (not a full test suite, but enough to show the
pipeline behaves as expected on edge cases). Run with: pytest test_pipeline.py
"""

import numpy as np
import pandas as pd

from generate_data import generate_patients
from clustering import build_preprocessor, run_kmeans, select_best_k, profile_clusters


def test_generate_patients_shape_and_ranges():
    df = generate_patients(n_patients=50, seed=1)
    assert len(df) == 50
    assert df["fever_severity"].between(0, 10).all()
    assert df["rash_present"].isin([0, 1]).all()
    assert set(df["employment_status"].unique()).issubset(
        {"employed_ft", "employed_pt", "unemployed", "disabled", "retired"}
    )


def test_preprocessor_output_is_numeric_and_finite():
    df = generate_patients(n_patients=40, seed=2).drop(columns=["patient_id", "true_archetype"])
    X = build_preprocessor().fit_transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    assert np.isfinite(X).all()
    assert X.shape[0] == 40


def test_kmeans_runs_and_produces_expected_number_of_clusters():
    df = generate_patients(n_patients=60, seed=3).drop(columns=["patient_id", "true_archetype"])
    X = build_preprocessor().fit_transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    result = run_kmeans(X, k=3)
    assert len(set(result.labels)) == 3
    assert -1 <= result.silhouette <= 1


def test_select_best_k_returns_value_in_range():
    df = generate_patients(n_patients=60, seed=4).drop(columns=["patient_id", "true_archetype"])
    X = build_preprocessor().fit_transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    best_k, scores = select_best_k(X, k_range=range(2, 5))
    assert 2 <= best_k <= 4
    assert len(scores) == 3


def test_profile_clusters_has_one_row_per_cluster():
    df = generate_patients(n_patients=60, seed=5)
    feature_df = df.drop(columns=["patient_id", "true_archetype"])
    X = build_preprocessor().fit_transform(feature_df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    result = run_kmeans(X, k=3)
    profile = profile_clusters(feature_df, result.labels)
    assert len(profile) == 3
    assert "n_patients" in profile.columns


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
