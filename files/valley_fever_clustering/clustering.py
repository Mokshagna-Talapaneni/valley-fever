"""
clustering.py
--------------
Clustering pipeline for grouping Valley Fever patients by symptom
presentation + social determinants of health (SDOH).

DESIGN CHOICES & ASSUMPTIONS
=============================
1. Feature scaling: Clinical severity scores (0-10), binary flags,
   continuous measures (weeks, miles, %) and SDOH scores are on very
   different scales. All numeric features are standardized (z-score)
   before clustering so that, e.g., "distance to clinic in miles" does
   not dominate "rash present (0/1)" purely because of scale.

2. Categorical encoding: employment_status, education_level,
   insurance_status, and income_bracket are one-hot encoded rather than
   label-encoded, since they are nominal/ordinal-but-unequally-spaced
   categories, not naturally continuous. One-hot avoids implying a false
   numeric distance between e.g. "employed_ft" and "unemployed".

3. Algorithm choice: K-Means is the primary method because (a) the
   feature space after one-hot + scaling is moderate-dimensional and
   roughly continuous, (b) K-Means is fast, deterministic given a seed,
   and easy to explain to a clinical/SDOH audience ("centroid = typical
   patient profile"), and (c) we have a working hypothesis of a small
   number of clinically meaningful subgroups (mild/moderate/severe),
   which suits a partition-based method with a chosen k. As a
   robustness check, Agglomerative (hierarchical) clustering is run on
   the same features and compared via the Adjusted Rand Index -- if the
   two very different algorithms roughly agree, that's evidence the
   structure is real rather than an artifact of K-Means' spherical-
   cluster assumption. Gaussian Mixture Models are also reported as a
   soft-clustering alternative since it relaxes the spherical-cluster
   assumption and gives cluster-membership probabilities, which can be
   clinically useful for patients who sit near a boundary.

4. Choosing k: Rather than assuming k=3, k is selected by scanning
   k=2..8 and picking the value that maximizes silhouette score (with
   the elbow in inertia reported alongside as a secondary signal). This
   keeps the method honest -- it should recover a sensible k on its own,
   not have the answer hard-coded to match how the synthetic data
   happened to be generated.

5. Dimensionality for visualization only: PCA to 2 components is used
   purely for plotting; clustering itself is always performed in the
   full standardized feature space, not the PCA-reduced space, so we
   don't lose information the algorithm could use.

6. Reproducibility: a fixed random_state is used throughout so the
   submitted screenshots are reproducible from the code as-is.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "fever_severity",
    "fatigue_severity",
    "cough_severity",
    "joint_pain_severity",
    "chest_pain_severity",
    "rash_present",
    "night_sweats_severity",
    "weight_loss_pct",
    "symptom_duration_weeks",
    "disseminated_disease",
    "housing_stability",
    "social_support_score",
    "access_to_care_score",
    "distance_to_clinic_mi",
]

CATEGORICAL_FEATURES = [
    "employment_status",
    "education_level",
    "insurance_status",
    "income_bracket",
]


def build_preprocessor() -> ColumnTransformer:
    """Standardize numeric features, one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="if_binary"), CATEGORICAL_FEATURES),
        ]
    )


@dataclass
class ClusteringResult:
    k: int
    labels: np.ndarray
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    model_name: str


def select_best_k(
    X: np.ndarray, k_range=range(2, 9), model_name: str = "kmeans"
) -> tuple[int, list[dict]]:
    """Scan candidate k values, score each with silhouette/DB/CH, return best k."""
    scores = []
    for k in k_range:
        if model_name == "kmeans":
            model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        elif model_name == "agglomerative":
            model = AgglomerativeClustering(n_clusters=k)
        else:
            raise ValueError(model_name)

        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        inertia = model.inertia_ if hasattr(model, "inertia_") else np.nan
        scores.append(
            {"k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch, "inertia": inertia}
        )

    best = max(scores, key=lambda s: s["silhouette"])
    return best["k"], scores


def run_kmeans(X: np.ndarray, k: int) -> ClusteringResult:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)
    return ClusteringResult(
        k=k,
        labels=labels,
        silhouette=silhouette_score(X, labels),
        davies_bouldin=davies_bouldin_score(X, labels),
        calinski_harabasz=calinski_harabasz_score(X, labels),
        model_name="KMeans",
    )


def run_agglomerative(X: np.ndarray, k: int) -> ClusteringResult:
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    return ClusteringResult(
        k=k,
        labels=labels,
        silhouette=silhouette_score(X, labels),
        davies_bouldin=davies_bouldin_score(X, labels),
        calinski_harabasz=calinski_harabasz_score(X, labels),
        model_name="Agglomerative",
    )


def run_gmm(X: np.ndarray, k: int) -> tuple[ClusteringResult, np.ndarray]:
    """Gaussian Mixture Model - also returns soft cluster-membership probabilities.

    covariance_type='tied' (one shared covariance matrix across all
    components, rather than one full covariance matrix per component) is
    used deliberately: after one-hot encoding we have ~30 dimensions but
    only a few hundred patients (and far fewer per cluster), so a
    separate full covariance per component is under-determined and
    produces badly overconfident, poorly-calibrated probabilities (we
    verified this empirically -- even a point exactly equidistant between
    two cluster centroids was scored at ~100% confidence with
    covariance_type='full'). 'tied' pools the covariance estimate across
    clusters, which is far more stable at this sample size and gives
    sensible, well-calibrated probabilities (the same equidistant point
    scores close to 50/50, as it should)."""
    model = GaussianMixture(
        n_components=k, random_state=RANDOM_STATE, n_init=5, covariance_type="tied", reg_covar=1e-3
    )
    labels = model.fit_predict(X)
    probs = model.predict_proba(X)
    result = ClusteringResult(
        k=k,
        labels=labels,
        silhouette=silhouette_score(X, labels),
        davies_bouldin=davies_bouldin_score(X, labels),
        calinski_harabasz=calinski_harabasz_score(X, labels),
        model_name="GaussianMixture",
    )
    return result, probs


def agreement_with_kmeans(kmeans_labels: np.ndarray, other_labels: np.ndarray) -> float:
    """Adjusted Rand Index between two label sets - robustness check."""
    return adjusted_rand_score(kmeans_labels, other_labels)


def pca_2d(X: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Build a human-readable summary table: mean of numeric features and
    mode of categorical features, per cluster, plus cluster size."""
    work = df.copy()
    work["cluster"] = labels

    numeric_summary = work.groupby("cluster")[NUMERIC_FEATURES].mean().round(2)

    cat_summary = pd.DataFrame(index=numeric_summary.index)
    for col in CATEGORICAL_FEATURES:
        mode_per_cluster = work.groupby("cluster")[col].agg(lambda s: s.mode().iat[0])
        cat_summary[col + "_mode"] = mode_per_cluster

    size = work.groupby("cluster").size().rename("n_patients")

    profile = pd.concat([size, numeric_summary, cat_summary], axis=1)
    return profile
