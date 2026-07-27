"""
feature_importance.py
-----------------------
Explains WHICH features actually drive the cluster split, rather than
just reporting that a split exists.

METHOD
======
Clustering is unsupervised, so there's no ground-truth label to compute
"importance" against directly. The standard workaround: treat the
cluster assignments themselves as a target and train a supervised
classifier (Random Forest) to predict cluster label from the original
features. The classifier's feature importances then tell you which
features the clusters actually differ on -- this is purely descriptive
(explaining the clustering, not validating it), but it's exactly what a
domain expert will ask first: "OK, you found two groups -- what actually
separates them?"

A high classifier accuracy is also a secondary sanity check: if the
clusters can't be predicted from the original features at all, something
is wrong with the clustering (e.g. it picked up on preprocessing
artifacts rather than real feature differences).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42


def compute_feature_importance(
    feature_df: pd.DataFrame, labels: np.ndarray, numeric_features: list, categorical_features: list
):
    """Train a Random Forest to predict cluster label from the *original*
    (human-readable, not one-hot/scaled) features and return sorted
    importances plus cross-validated accuracy."""
    X = pd.get_dummies(feature_df[numeric_features + categorical_features], drop_first=False)
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, max_depth=6)

    cv_acc = cross_val_score(clf, X, labels, cv=5, scoring="accuracy")
    clf.fit(X, labels)

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return importances, cv_acc
