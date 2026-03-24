import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


# Load data
csv_path = "/home/feketeova/Documents/results.csv"
df = pd.read_csv(csv_path, header=0)

# First column = patient IDs, second column = skipped, rest = features
patient_ids = df.iloc[:, 0].astype(str).values
features    = df.columns[2:]




def within_cluster_ss(data, labels):
    return sum(np.var(data[labels == k]) for k in np.unique(labels))


def permutation_test(data, k=2, n_perm=1000):
    Z           = linkage(data.reshape(-1, 1), method='ward')
    labels      = fcluster(Z, k, criterion='maxclust')
    real_score  = within_cluster_ss(data, labels)

    perm_scores = []
    for _ in range(n_perm):
        shuffled    = np.random.permutation(data)
        Z_perm      = linkage(shuffled.reshape(-1, 1), method='ward')
        labels_perm = fcluster(Z_perm, k, criterion='maxclust')
        perm_scores.append(within_cluster_ss(shuffled, labels_perm))

    p_value = np.mean(np.array(perm_scores) <= real_score)
    return p_value


# Main

results = []

for feature in features:
    print(f"\n=== Analyzing: {feature} ===")

    # Keep only rows where this feature is not NaN, carry patient IDs along
    mask        = df[feature].notna()
    data        = df.loc[mask, feature].values
    ids_feature = patient_ids[mask]

    if len(data) < 5:
        print("Too few samples, skipping")
        continue

    data_reshaped = data.reshape(-1, 1)

    # Hierarchical clustering
    Z = linkage(data_reshaped, method='ward')

    # Leaf order from dendrogram (needed to map patient IDs to x-axis positions)
    ddata      = dendrogram(Z, no_plot=True)
    leaf_order = ddata['leaves']             # indices in dendrogram left→right order
    id_labels  = ids_feature[leaf_order]     # patient IDs in that order

    # Finding k via silhouette
    sil_scores = {}
    for k in range(2, min(6, len(data))):
        labels = fcluster(Z, k, criterion='maxclust')
        try:
            sil_scores[k] = silhouette_score(data_reshaped, labels)
        except Exception:
            sil_scores[k] = np.nan

    best_k_sil = max(
        sil_scores,
        key=lambda k: sil_scores[k] if not np.isnan(sil_scores[k]) else -1
    )

    # GMM BIC
    bics = {}
    for k in range(1, min(6, len(data))):
        gmm = GaussianMixture(n_components=k, random_state=0)
        gmm.fit(data_reshaped)
        bics[k] = gmm.bic(data_reshaped)

    best_k_bic = min(bics, key=bics.get)

    # Permutation test
    k_test = max(best_k_sil, 2)
    p_perm = permutation_test(data, k=k_test, n_perm=500)

    # Cluster assignments at best_k_sil for the scatter plot
    labels_best = fcluster(Z, best_k_sil, criterion='maxclust')

    print(f"  Best K silhouette={best_k_sil}  GMM BIC={best_k_bic}  perm p={p_perm:.4f}")

    # Store results — include per-patient values and cluster assignments
    for i, (pid, val, clust) in enumerate(zip(ids_feature, data, labels_best)):
        results.append({
            "patient_id":           pid,
            "feature":              feature,
            "value":                round(float(val), 6),
            "cluster":              int(clust),
            "best_k_silhouette":    best_k_sil,
            "best_k_gmm_bic":       best_k_bic,
            "permutation_p_value":  round(float(p_perm), 4),
        })

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Histogram
    axes[0].hist(data, bins=10, color='steelblue', edgecolor='white')
    axes[0].set_title(f"{feature}\nHistogram")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")

    # 2. Dendrogram with patient IDs on x-axis and values as subtitle
    ax_dendro = axes[1]
    dendrogram(
        Z,
        labels=ids_feature,        # patient IDs as leaf labels
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax_dendro,
        color_threshold=Z[-(best_k_sil - 1), 2] if best_k_sil > 1 else 0,
    )
    ax_dendro.set_title(f"{feature}\nDendrogram (Ward)  K={best_k_sil}")
    ax_dendro.set_xlabel("Patient ID")
    ax_dendro.set_ylabel("Distance")

    # 3. Scatter of values per patient, coloured by cluster
    colors   = plt.cm.tab10(np.linspace(0, 1, best_k_sil))
    sort_idx = np.argsort(labels_best)   # group by cluster for readability
    for k in range(1, best_k_sil + 1):
        mask_k = labels_best == k
        axes[2].scatter(
            ids_feature[mask_k],
            data[mask_k],
            label=f"Cluster {k}",
            color=colors[k - 1],
            s=60, zorder=3
        )
    axes[2].set_title(f"{feature}\nValues by patient  (p={p_perm:.3f})")
    axes[2].set_xlabel("Patient ID")
    axes[2].set_ylabel("Value")
    axes[2].tick_params(axis='x', rotation=90, labelsize=7)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"plot_{feature.replace('/', '_').replace(' ', '_')}.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("clustering_results.csv", index=False)
print("\nDone. Results saved to clustering_results.csv")