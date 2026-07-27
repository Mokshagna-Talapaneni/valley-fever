"""
stability.py
-------------
Bootstrap stability analysis for the chosen clustering.

WHY THIS MATTERS
================
Silhouette/DB/CH scores tell you how well-separated a *single* clustering
run is, but they say nothing about whether you'd get roughly the same
groups if you had a slightly different sample of patients (e.g. a
different month's intake, or a different clinic). A clustering that
shifts wildly on resampling is not something you'd want to base care
coordination decisions on, however good its silhouette score looks.

METHOD
======
Standard bootstrap stability procedure:
  1. Fit the reference clustering (KMeans, k) on the full dataset.
  2. Draw B bootstrap resamples (sampling with replacement, same size as
     original).
  3. Re-fit KMeans(k) on each resample.
  4. For each resample, restrict the reference labels to the same
     (index, with-duplicates) rows and compare to the new clustering via
     Adjusted Rand Index.
  5. Report the mean +/- std ARI across B resamples as the stability
     score. As a rule of thumb: >0.75 = stable, 0.5-0.75 = moderately
     stable, <0.5 = unstable / don't trust the grouping.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

RANDOM_STATE = 42


def bootstrap_stability(X: np.ndarray, k: int, n_boot: int = 50, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    reference = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
    reference_labels = reference.labels_

    aris = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        X_boot = X[idx]
        boot_labels = KMeans(n_clusters=k, random_state=seed + b, n_init=10).fit_predict(X_boot)
        ref_labels_boot = reference_labels[idx]  # reference labels for the same rows
        aris.append(adjusted_rand_score(ref_labels_boot, boot_labels))

    aris = np.array(aris)
    return {
        "k": k,
        "n_boot": n_boot,
        "mean_ari": float(aris.mean()),
        "std_ari": float(aris.std()),
        "min_ari": float(aris.min()),
        "max_ari": float(aris.max()),
        "all_ari": aris,
    }


def interpret_stability(mean_ari: float) -> str:
    if mean_ari > 0.75:
        return "stable"
    elif mean_ari > 0.5:
        return "moderately stable"
    else:
        return "unstable - interpret groupings cautiously"
