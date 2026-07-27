"""
main.py
-------
End-to-end run: generate synthetic data -> preprocess -> select k ->
cluster (KMeans, cross-checked with Agglomerative + GMM) -> evaluate ->
profile clusters -> save plots and summary tables to outputs/.

Run with:  python main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from generate_data import generate_patients
from clustering import (
    build_preprocessor,
    select_best_k,
    run_kmeans,
    run_agglomerative,
    run_gmm,
    agreement_with_kmeans,
    pca_2d,
    profile_clusters,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from stability import bootstrap_stability, interpret_stability
from feature_importance import compute_feature_importance
from equity_audit import audit_categorical_association, audit_numeric_association

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    # ---- 1. Data ----------------------------------------------------
    df = generate_patients(n_patients=300)
    df.to_csv(f"{OUT_DIR}/synthetic_patients.csv", index=False)
    print(f"[1/8] Generated {len(df)} synthetic patients.")

    feature_df = df.drop(columns=["patient_id", "true_archetype"])

    # ---- 2. Preprocess ------------------------------------------------
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(feature_df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    print(f"[2/8] Preprocessed feature matrix shape: {X.shape}")

    # ---- 3. Select k via silhouette scan ------------------------------
    best_k, k_scores = select_best_k(X, k_range=range(2, 9), model_name="kmeans")
    scores_df = pd.DataFrame(k_scores)
    scores_df.to_csv(f"{OUT_DIR}/k_selection_scores.csv", index=False)
    print(f"[3/8] Best k by silhouette score: {best_k}")
    print(scores_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(scores_df["k"], scores_df["silhouette"], marker="o")
    axes[0].axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"chosen k={best_k}")
    axes[0].set_title("Silhouette score vs k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silhouette score")
    axes[0].legend()

    axes[1].plot(scores_df["k"], scores_df["inertia"], marker="o", color="darkorange")
    axes[1].set_title("Elbow plot (inertia vs k)")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_k_selection.png", dpi=150)
    plt.close()

    # ---- 4. Cluster with KMeans, cross-check with Agglomerative + GMM -
    kmeans_result = run_kmeans(X, best_k)
    agg_result = run_agglomerative(X, best_k)
    gmm_result, gmm_probs = run_gmm(X, best_k)

    ari_agg = agreement_with_kmeans(kmeans_result.labels, agg_result.labels)
    ari_gmm = agreement_with_kmeans(kmeans_result.labels, gmm_result.labels)

    print(f"[4/8] KMeans   silhouette={kmeans_result.silhouette:.3f}  "
          f"DB={kmeans_result.davies_bouldin:.3f}  CH={kmeans_result.calinski_harabasz:.1f}")
    print(f"      Agglom.  silhouette={agg_result.silhouette:.3f}  "
          f"DB={agg_result.davies_bouldin:.3f}  CH={agg_result.calinski_harabasz:.1f}  "
          f"(ARI vs KMeans={ari_agg:.3f})")
    print(f"      GMM      silhouette={gmm_result.silhouette:.3f}  "
          f"DB={gmm_result.davies_bouldin:.3f}  CH={gmm_result.calinski_harabasz:.1f}  "
          f"(ARI vs KMeans={ari_gmm:.3f})")

    # How well does the unsupervised clustering recover the hidden
    # ground-truth archetype used to generate the data? (internal
    # validation only -- would not exist for real unlabeled data)
    true_labels = df["true_archetype"].astype("category").cat.codes.values
    ari_vs_truth = agreement_with_kmeans(true_labels, kmeans_result.labels)
    print(f"      ARI of KMeans clusters vs hidden synthetic archetype: {ari_vs_truth:.3f}")

    algo_comparison = pd.DataFrame(
        [
            {"model": "KMeans", "k": kmeans_result.k, "silhouette": kmeans_result.silhouette,
             "davies_bouldin": kmeans_result.davies_bouldin, "calinski_harabasz": kmeans_result.calinski_harabasz,
             "ARI_vs_KMeans": 1.0},
            {"model": "Agglomerative", "k": agg_result.k, "silhouette": agg_result.silhouette,
             "davies_bouldin": agg_result.davies_bouldin, "calinski_harabasz": agg_result.calinski_harabasz,
             "ARI_vs_KMeans": ari_agg},
            {"model": "GaussianMixture", "k": gmm_result.k, "silhouette": gmm_result.silhouette,
             "davies_bouldin": gmm_result.davies_bouldin, "calinski_harabasz": gmm_result.calinski_harabasz,
             "ARI_vs_KMeans": ari_gmm},
        ]
    )
    algo_comparison.to_csv(f"{OUT_DIR}/algorithm_comparison.csv", index=False)
    print(algo_comparison.to_string(index=False))

    # ---- 4b. Bootstrap stability of the chosen KMeans clustering -------
    stability = bootstrap_stability(X, best_k, n_boot=50)
    print(f"[4b/8] Bootstrap stability (k={best_k}, 50 resamples): "
          f"mean ARI={stability['mean_ari']:.3f} (+/-{stability['std_ari']:.3f}) "
          f"-> {interpret_stability(stability['mean_ari'])}")
    pd.DataFrame({"ari": stability["all_ari"]}).to_csv(f"{OUT_DIR}/bootstrap_stability_scores.csv", index=False)

    plt.figure(figsize=(5, 4))
    plt.hist(stability["all_ari"], bins=15, color="steelblue", edgecolor="white")
    plt.axvline(stability["mean_ari"], color="red", linestyle="--", label=f"mean={stability['mean_ari']:.2f}")
    plt.title(f"Bootstrap stability (k={best_k}, n={stability['n_boot']} resamples)")
    plt.xlabel("Adjusted Rand Index vs reference clustering")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_bootstrap_stability.png", dpi=150)
    plt.close()

    # ---- 4c. Feature importance: what actually drives the split? ------
    importances, cv_acc = compute_feature_importance(
        feature_df, kmeans_result.labels, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    print(f"[4c/8] Random-forest cluster-label classifier CV accuracy: "
          f"{cv_acc.mean():.3f} (+/-{cv_acc.std():.3f})")
    print("      Top 8 features driving the cluster split:")
    print(importances.head(8).to_string())
    importances.to_csv(f"{OUT_DIR}/feature_importance.csv", header=["importance"])

    plt.figure(figsize=(7, 5))
    importances.head(12).sort_values().plot(kind="barh", color="teal")
    plt.title("Top features separating the clusters\n(Random Forest importance, predicting cluster from features)")
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_feature_importance.png", dpi=150)
    plt.close()

    # ---- 4d. Equity audit: are clusters just SDOH proxies? -------------
    cat_audit = audit_categorical_association(feature_df, kmeans_result.labels, CATEGORICAL_FEATURES)
    sdoh_numeric = ["housing_stability", "social_support_score", "access_to_care_score", "distance_to_clinic_mi"]
    num_audit = audit_numeric_association(feature_df, kmeans_result.labels, sdoh_numeric)
    print("[4d/8] Equity audit -- association between cluster membership and SDOH variables:")
    print(cat_audit.to_string(index=False))
    print(num_audit.to_string(index=False))
    cat_audit.to_csv(f"{OUT_DIR}/equity_audit_categorical.csv", index=False)
    num_audit.to_csv(f"{OUT_DIR}/equity_audit_numeric.csv", index=False)

    # ---- 5. Visualize clusters (PCA 2D) --------------------------------
    coords = pca_2d(X)
    plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    plot_df["cluster"] = kmeans_result.labels.astype(str)
    plot_df["true_archetype"] = df["true_archetype"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster", palette="Set2", ax=axes[0], s=45)
    axes[0].set_title(f"KMeans clusters (k={best_k}) in PCA space")

    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="true_archetype", palette="Set1", ax=axes[1], s=45)
    axes[1].set_title("Hidden synthetic archetype (ground truth, for reference)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_cluster_scatter_pca.png", dpi=150)
    plt.close()

    # ---- 6. Cluster profiles -------------------------------------------
    profile = profile_clusters(feature_df, kmeans_result.labels)
    profile.to_csv(f"{OUT_DIR}/cluster_profiles.csv")
    print("[8/8] Cluster profiles:")
    print(profile.to_string())

    # Heatmap of standardized cluster means for quick visual comparison
    z_profile = (profile[NUMERIC_FEATURES] - profile[NUMERIC_FEATURES].mean()) / profile[NUMERIC_FEATURES].std()
    plt.figure(figsize=(11, 4.5))
    sns.heatmap(z_profile, annot=profile[NUMERIC_FEATURES].values, fmt=".1f", cmap="RdBu_r", center=0,
                cbar_kws={"label": "z-score across clusters"})
    plt.title("Cluster profile heatmap (annotated with raw mean values)")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_cluster_profile_heatmap.png", dpi=150)
    plt.close()

    # Cluster sizes bar chart
    plt.figure(figsize=(5, 4))
    profile["n_patients"].plot(kind="bar", color=sns.color_palette("Set2"))
    plt.title("Patients per cluster")
    plt.xlabel("Cluster")
    plt.ylabel("n_patients")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_cluster_sizes.png", dpi=150)
    plt.close()

    print("\nAll outputs written to ./outputs/")


if __name__ == "__main__":
    main()
