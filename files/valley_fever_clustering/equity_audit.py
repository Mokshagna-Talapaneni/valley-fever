"""
equity_audit.py
-----------------
Checks whether the resulting clusters are, in effect, just re-encoding
socioeconomic proxies (income, insurance status) rather than capturing
something clinically distinct -- and flags that risk explicitly rather
than leaving it implicit in a heatmap.

WHY THIS MATTERS
================
This pipeline deliberately includes SDOH features (housing, income,
insurance, employment) alongside clinical symptoms, because that's the
whole point of the assignment -- SDOH is often what determines whether a
patient can get timely care. But that also means the resulting clusters
could end up being driven almost entirely by socioeconomic status, with
symptom severity riding along as a correlated side effect (worse access
to care -> later diagnosis -> more severe symptoms by the time they're
seen). If a "high-risk" cluster is then used to allocate resources or
flag patients, it needs to be understood as, and communicated as, a
resource/access risk group -- not mistaken for a purely clinical severity
group, and its use needs to be checked against the risk of driving
inequitable downstream decisions (e.g., denying care intensity to
patients whose access is already limited, rather than proactively
supporting them).

This script does NOT resolve that tension -- that's a policy and clinical
governance question -- but it does quantify how strongly cluster
membership associates with each SDOH variable, so the tension is visible
and can be discussed with domain experts / an ethics review before any
real-world use.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score


def audit_categorical_association(feature_df: pd.DataFrame, labels: np.ndarray, categorical_features: list):
    """For each categorical SDOH feature, report a Cramer's V association
    strength with cluster membership (0 = no association, 1 = perfect
    association) via a chi-square test."""
    results = []
    for col in categorical_features:
        contingency = pd.crosstab(feature_df[col], labels)
        chi2, p, dof, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
        mi = mutual_info_score(feature_df[col], labels)
        results.append({"feature": col, "cramers_v": round(cramers_v, 3), "p_value": p, "mutual_info": round(mi, 3)})
    return pd.DataFrame(results).sort_values("cramers_v", ascending=False)


def audit_numeric_association(feature_df: pd.DataFrame, labels: np.ndarray, numeric_sdoh_features: list):
    """For numeric SDOH features (e.g. housing_stability, social_support,
    access_to_care, distance), report the between-cluster mean gap in
    standard-deviation units (Cohen's d style effect size, generalized to
    multi-cluster via max pairwise gap)."""
    results = []
    df = feature_df.copy()
    df["cluster"] = labels
    for col in numeric_sdoh_features:
        overall_std = df[col].std()
        means = df.groupby("cluster")[col].mean()
        max_gap = (means.max() - means.min()) / overall_std if overall_std > 0 else np.nan
        results.append({"feature": col, "max_standardized_gap_between_clusters": round(max_gap, 2)})
    return pd.DataFrame(results).sort_values("max_standardized_gap_between_clusters", ascending=False)
