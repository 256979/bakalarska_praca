import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


# Load data

csv_path = "/home/feketeova/Documents/results.csv"
df = pd.read_csv(csv_path, header=0)



# Assume first two columns are IDs → skip them
features = df.columns[2:]



# Helper functions

def within_cluster_ss(data, labels):
    return sum(np.var(data[labels == k]) for k in np.unique(labels))


def permutation_test(data, k=2, n_perm=1000):
    Z = linkage(data.reshape(-1, 1), method='ward')
    labels = fcluster(Z, k, criterion='maxclust')
    real_score = within_cluster_ss(data, labels)

    perm_scores = []
    for _ in range(n_perm):
        shuffled = np.random.permutation(data)
        Z_perm = linkage(shuffled.reshape(-1, 1), method='ward')
        labels_perm = fcluster(Z_perm, k, criterion='maxclust')
        perm_scores.append(within_cluster_ss(shuffled, labels_perm))

    p_value = np.mean(np.array(perm_scores) <= real_score)
    return p_value



# Main

results = []

for feature in features:
    print(f"\n=== Analyzing: {feature} ===")

    data = df[feature].dropna().values

    # Skip tiny features
    if len(data) < 5:
        print("Too few samples, skipping")
        continue

    data_reshaped = data.reshape(-1, 1)


    # Hierarchical clustering

    Z = linkage(data_reshaped, method='ward')


    # Finding k (support of visual dendogram cutting)

    sil_scores = {}
    for k in range(2, min(6, len(data))):
        labels = fcluster(Z, k, criterion='maxclust')
        try:
            score = silhouette_score(data_reshaped, labels)
            sil_scores[k] = score
        except:
            sil_scores[k] = np.nan

    best_k_sil = max(sil_scores, key=lambda k: sil_scores[k] if not np.isnan(sil_scores[k]) else -1)


    # GMM

    bics = {}
    for k in range(1, min(6, len(data))):
        gmm = GaussianMixture(n_components=k)
        gmm.fit(data_reshaped)
        bics[k] = gmm.bic(data_reshaped)

    best_k_bic = min(bics, key=bics.get)




    # Permutation test

    k_test = max(best_k_sil, 2)
    p_perm = permutation_test(data, k=k_test, n_perm=500)


    # Storing results

    results.append({
        "feature": feature,
        "best_k_silhouette": best_k_sil,
        "best_k_gmm_bic": best_k_bic,

        "permutation_p_value": p_perm
    })


    # Graphs

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

    plt.subplot(1, 2, 1)
    plt.hist(data, bins=10)
    plt.title(f"{feature} histogram")

    plt.subplot(1, 2, 2)
    dendrogram(Z)
    plt.title(f"{feature} dendrogram")

    plt.tight_layout()
    plt.show()


# Saving results
results_df = pd.DataFrame(results)
results_df.to_csv("clustering_results.csv", index=False)