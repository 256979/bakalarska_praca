import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score


# =========================================================
# Create folder for plots
# =========================================================
plot_dir = "plots_all_k"
os.makedirs(plot_dir, exist_ok=True)


# Load data
csv_path = "/home/feketeova/Documents/results.csv"
df = pd.read_csv(csv_path, header=0)

# First column = patient IDs, second column = skipped, rest = features
patient_ids = df.iloc[:, 0].astype(str).values
features    = df.columns[2:]


# =========================================================
# Helper functions
# =========================================================

def within_cluster_ss(data, labels):
    """Sum of within-cluster variances."""
    return sum(np.var(data[labels == k]) for k in np.unique(labels))


def improved_permutation_test(data, k=2, n_perm=300):
    """
    Improved permutation test:
    - Uses Ward linkage (same as real clustering)
    - Uses variance-based cluster quality
    - More stable for small sample sizes
    """
    data = np.asarray(data)
    Z = linkage(data.reshape(-1, 1), method='ward')
    labels = fcluster(Z, k, criterion='maxclust')
    real_score = within_cluster_ss(data, labels)

    perm_scores = np.zeros(n_perm)

    for i in range(n_perm):
        shuffled = np.random.permutation(data)
        Z_perm = linkage(shuffled.reshape(-1, 1), method='ward')
        labels_perm = fcluster(Z_perm, k, criterion='maxclust')
        perm_scores[i] = within_cluster_ss(shuffled, labels_perm)

    # p-value = fraction of permutations that produce equal or better clustering
    return float(np.mean(perm_scores <= real_score))


# =========================================================
# Main
# =========================================================

results_all_k = []            # per-patient cluster assignments
results_best_only = []        # best k per feature
significant_results = []      # significant features
stats_per_feature_k = []      # one row per feature per k


for feature in features:
    print(f"\n=== Analyzing: {feature} ===")

    # Keep only rows where this feature is not NaN
    mask        = df[feature].notna()
    data        = df.loc[mask, feature].values
    ids_feature = patient_ids[mask]

    # =========================================================
    # Minimum sample requirement = 2
    # =========================================================
    if len(data) < 2:
        print("Too few samples (<2), skipping")
        continue

    data_reshaped = data.reshape(-1, 1)

    # Hierarchical clustering
    Z = linkage(data_reshaped, method='ward')

    # Try all k from 2 to 5 (or max possible)
    max_k = min(6, len(data))
    sil_scores = {}

    for k in range(2, max_k):

        labels_k = fcluster(Z, k, criterion='maxclust')

        # =========================================================
        # Silhouette with fallback
        # =========================================================
        try:
            sil = silhouette_score(data_reshaped, labels_k)
        except:
            sil = 0.0  # fallback instead of NaN

        sil_scores[k] = sil

        # =========================================================
        # Improved permutation test
        # =========================================================
        p_perm_k = improved_permutation_test(data, k=k, n_perm=300)

        # =========================================================
        # Store statistics ONCE per feature per k
        # =========================================================
        stats_per_feature_k.append({
            "feature": feature,
            "k": k,
            "silhouette": sil,
            "permutation_p_value": round(p_perm_k, 4),
        })

        # =========================================================
        # Store per-patient cluster assignments
        # =========================================================
        for pid, val, clust in zip(ids_feature, data, labels_k):
            results_all_k.append({
                "patient_id": pid,
                "feature": feature,
                "value": round(float(val), 6),
                "cluster": int(clust),
                "k": k,
            })

        # =========================================================
        # Store significant results
        # =========================================================
        if p_perm_k < 0.05:
            significant_results.append({
                "feature": feature,
                "k": k,
                "p_value": round(p_perm_k, 4),
                "silhouette": sil,
            })

        ''' # =========================================================
        # PLOTS
        # =========================================================
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. Histogram
        axes[0].hist(data, bins=10, color='steelblue', edgecolor='white')
        axes[0].set_title(f"{feature} (k={k})\nHistogram")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Count")

        # 2. Dendrogram
        dendrogram(
            Z,
            labels=ids_feature,
            leaf_rotation=90,
            leaf_font_size=7,
            ax=axes[1],
        )
        axes[1].set_title(f"{feature} (k={k})\nDendrogram (Ward)")
        axes[1].set_xlabel("Patient ID")
        axes[1].set_ylabel("Distance")

        # 3. Scatter of values per patient
        colors = plt.cm.tab10(np.linspace(0, 1, k))
        for kk in range(1, k + 1):
            mask_k = labels_k == kk
            axes[2].scatter(
                ids_feature[mask_k],
                data[mask_k],
                label=f"Cluster {kk}",
                color=colors[kk - 1],
                s=60, zorder=3
            )
        axes[2].set_title(f"{feature} (k={k})\nValues by patient")
        axes[2].set_xlabel("Patient ID")
        axes[2].set_ylabel("Value")
        axes[2].tick_params(axis='x', rotation=90, labelsize=7)
        axes[2].legend(fontsize=8)

        plt.tight_layout()

        safe_name = feature.replace("/", "_").replace(" ", "_")
        plt.savefig(
            os.path.join(plot_dir, f"plot_{safe_name}_k{k}.png"),
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()'''

    # =========================================================
    # Determine best k (silhouette only)
    # =========================================================
    best_k_sil = max(sil_scores, key=lambda k: sil_scores[k])

    labels_best = fcluster(Z, best_k_sil, criterion='maxclust')

    # Store best-k results
    for pid, val, clust in zip(ids_feature, data, labels_best):
        results_best_only.append({
            "patient_id": pid,
            "feature": feature,
            "value": round(float(val), 6),
            "cluster": int(clust),
            "best_k_silhouette": best_k_sil,
        })


# =========================================================
# Save results
# =========================================================

pd.DataFrame(results_all_k).to_csv("clustering_all_k_new.csv", index=False)
pd.DataFrame(results_best_only).to_csv("clustering_best_k_new.csv", index=False)
pd.DataFrame(significant_results).to_csv("clustering_significant_only_new.csv", index=False)
pd.DataFrame(stats_per_feature_k).to_csv("clustering_stats_per_feature_k_new.csv", index=False)

print("\nSaved:")
print(" - clustering_all_k.csv")
print(" - clustering_best_k.csv")
print(" - clustering_significant_only.csv")
print(" - clustering_stats_per_feature_k.csv")
print(f" - All plots saved in folder: {plot_dir}/")
