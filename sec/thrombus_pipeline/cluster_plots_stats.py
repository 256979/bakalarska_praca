import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
from scipy.stats import shapiro, ttest_ind, mannwhitneyu


def load_feature_csv(path):
    df = pd.read_csv(path)
    patient_ids = df.iloc[:, 0].astype(str).values
    features = df.columns[1:]
    return df, patient_ids, features


def cluster_feature(data, k=2):
    Z = linkage(data.reshape(-1, 1), method="ward")
    labels = fcluster(Z, k, criterion="maxclust")

    sil = silhouette_score(data.reshape(-1, 1), labels) \
        if len(np.unique(labels)) > 1 else np.nan

    return Z, labels, sil


def check_split_is_two(Z):
    second_last_dist = Z[-2, 2]
    labels = fcluster(Z, second_last_dist, criterion="distance")
    return len(np.unique(labels)) == 2


def compute_statistics(data, labels):
    c1 = data[labels == 1]
    c2 = data[labels == 2]

    n1, n2 = len(c1), len(c2)
    mean1, mean2 = np.mean(c1), np.mean(c2)

    p_norm1 = shapiro(c1)[1] if n1 >= 3 else np.nan
    p_norm2 = shapiro(c2)[1] if n2 >= 3 else np.nan

    if (
        not np.isnan(p_norm1) and not np.isnan(p_norm2)
        and p_norm1 >= 0.05 and p_norm2 >= 0.05
    ):
        test_used = "Welch t-test"
        stat, p_val = ttest_ind(c1, c2, equal_var=False)
    else:
        test_used = "Mann-Whitney U"
        stat, p_val = mannwhitneyu(c1, c2, alternative="two-sided")

    return {
        "n_cluster1": n1,
        "n_cluster2": n2,
        "mean_cluster1": mean1,
        "mean_cluster2": mean2,
        "p_normality_cluster1": p_norm1,
        "p_normality_cluster2": p_norm2,
        "test_used": test_used,
        "test_statistic": stat,
        "p_value": p_val
    }


def plot_dendrogram(lin_m, patient_ids, feature, ax):
    dendrogram(
        lin_m,
        labels=patient_ids,
        leaf_rotation=90,
        leaf_font_size=28,
        ax=ax,
    )
    ax.set_title(f"{feature}\nDendrogram (Ward)", fontsize=30)
    ax.set_xlabel("Patient ID", fontsize=30)
    ax.set_ylabel("Distance", fontsize=30)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)


def plot_scatter(data, labels, patient_ids, feature, k, ax):
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    for kk in range(1, k + 1):
        mask = labels == kk
        ax.scatter(
            patient_ids[mask],
            data[mask],
            label=f"Cluster {kk}",
            color=colors[kk - 1],
            s=200, zorder=3
        )
    ax.set_title(f"{feature} (k={k})\nValues by patient", fontsize=30)
    ax.set_xlabel("Patient ID", fontsize=30)
    ax.set_ylabel("Value", fontsize=30)
    ax.tick_params(axis='x', rotation=90, labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.legend(fontsize=28)


def save_plots(lin_m, data, labels, patient_ids, feature, plot_dir, k):
    fig, axes = plt.subplots(1, 2, figsize=(30, 15))

    plot_dendrogram(lin_m, patient_ids, feature, axes[0])
    plot_scatter(data, labels, patient_ids, feature, k, axes[1])

    plt.tight_layout()

    safe_name = feature.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(plot_dir, f"plot_{safe_name}_k{k}.png")

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path


def process_feature(df, patient_ids, feature, plot_dir, config, k=2):
    print(f"Processing feature: {feature}")

    data = df[feature].values.astype(float)

    if np.isnan(data).any():
        print(" → Skipped (contains NaN)")
        return None
    if np.all(data == data[0]):
        print(" → Skipped (constant values)")
        return None

    Z, labels, sil = cluster_feature(data, k)
    root_is_two = check_split_is_two(Z)
    stats = compute_statistics(data, labels)
    p_val = stats["p_value"]
    n_clusters = len(np.unique(labels))

    sil_threshold = config["statistics"]["silhouette_threshold"]
    p_threshold = config["statistics"]["p_value_threshold"]

    is_significant = (
        n_clusters == 2 and
        sil is not None and sil >= sil_threshold and
        p_val < p_threshold and
        root_is_two
    )

    sign_value = 1 if is_significant else 0

    plot_path = save_plots(Z, data, labels, patient_ids, feature, plot_dir, k)
    print(f" → Saved: {plot_path}")

    return {
        "feature": feature,
        "silhouette": sil,
        "n_clusters": n_clusters,
        "p_value": p_val,
        "root_split_two": root_is_two,
        **stats,
        "sign": sign_value
    }


def process_all_features(df, patient_ids, features, plot_dir, config, k=2):
    stats_rows = []
    for feature in features:
        row = process_feature(df, patient_ids, feature, plot_dir, config, k)
        if row is not None:
            stats_rows.append(row)
    return stats_rows
