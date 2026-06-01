import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform




# LOAD DATA

'''
patient_values_path: str - path to the patient feature CSV (from config.toml)
sign_path: str – path to CSV containing feature significance table (from config.toml)

This function:
    Loads the patient-feature matrix.
    Loads the feature-sign table.
    Extracts the first column as feature name and last column as sign.
    Returns:
        df_feat – patient-feature matrix
        df_sign – feature-sign table with columns ["feature", "sign"]
'''
def load_data(patient_values_path, sign_path):

    df_feat = pd.read_csv(patient_values_path)

    if "Patient" not in df_feat.columns:
        raise ValueError("Missing 'Patient' column in patient values file.")

    df_sign_raw = pd.read_csv(sign_path)
    feature_col = df_sign_raw.columns[0]
    sign_col = df_sign_raw.columns[-1]

    df_sign = df_sign_raw[[feature_col, sign_col]].copy()
    df_sign.columns = ["feature", "sign"]

    return df_feat, df_sign




# SELECT SIGNIFICANT FEATURES

'''
df_feat: DataFrame – patient-feature matrix
df_sign: DataFrame – feature-sign table (loaded using paths from config.toml)

This function:
    Selects only features with sign=1.
    Ensures they exist in the patient-feature matrix.
    Returns:
        df_feat_filtered – patient-feature matrix with only significant features
        significant – list of significant feature names
'''
def select_significant_features(df_feat, df_sign):

    significant = df_sign.loc[df_sign["sign"] == 1, "feature"].tolist()
    significant = [f for f in significant if f in df_feat.columns]

    if len(significant) == 0:
        raise ValueError("No sign=1 features found in patient matrix.")

    df_feat = df_feat[["Patient"] + significant].copy()
    return df_feat, significant



# 3. PATIENT CLUSTERING (WARD)

'''
df_feat: DataFrame – patient-feature matrix (significant features only)
significant: list – names of significant features
out_dir: str – output directory (from config.toml)

This function:
    Performs Ward hierarchical clustering on patients using only significant features.
    Saves:
        - patient cluster assignments
        - dendrogram of patients
    Returns:
        labels – cluster labels for each patient
'''
def cluster_patients(df_feat, significant, out_dir):

    X = df_feat[significant].copy()
    X = X.loc[:, X.var() > 0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z = linkage(X_scaled, method="ward")
    labels = fcluster(Z, 2, criterion="maxclust")

    pd.DataFrame({
        "Patient": df_feat["Patient"],
        "cluster": labels
    }).to_csv(os.path.join(out_dir, "patient_clusters_ward_sign_1.csv"), index=False)

    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=df_feat["Patient"].values, leaf_rotation=90)
    plt.title("Patient clustering from significant features (Ward linkage)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dendrogram_ward_sign_1.png"), dpi=300)
    plt.close()

    return labels



# 4. FEATURE CORRELATION + DENDROGRAM (UPDATED)

'''
df_feat: DataFrame – patient-feature matrix (significant features only)
significant: list – names of significant features
out_dir: str – output directory (from config.toml)

This function:
    Computes correlation matrix between significant features.
    Saves:
        - correlation CSV
        - LOWER-HALF-ONLY correlation heatmap (fullscreen)
        - feature dendrogram
'''
def analyze_features(df_feat, significant, out_dir):


    corr = df_feat[significant].corr()
    corr.to_csv(os.path.join(out_dir, "feature_feature_correlation_matrix_sign_1.csv"))


    corr = corr.loc[significant, significant]


    lower = corr.copy()
    for i in range(len(lower)):
        for j in range(len(lower)):
            if j > i:
                lower.iat[i, j] = np.nan


    plt.figure(figsize=(70, 60), dpi=100)

    im = plt.imshow(lower, cmap="coolwarm", vmin=-1, vmax=1)

    plt.grid(which="both", color="white", linewidth=2.0)

    plt.xticks(range(len(significant)), significant, rotation=90, fontsize=38)
    plt.yticks(range(len(significant)), significant, fontsize=38)

    ax = plt.gca()
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(8)

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=36)
    cbar.set_label("Correlation", fontsize=60)

    plt.title("Feature–Feature Correlation Matrix (Lower Half Only)", fontsize=60)


    plt.subplots_adjust(left=0.18, bottom=0.38, right=0.98, top=0.92)

    plt.savefig(os.path.join(out_dir, "feature_feature_corr_lower_half_sign_1.png"), dpi=150)
    plt.close()


    corr_dist = 1 - np.abs(corr.values)
    condensed = squareform(corr_dist, checks=False)

    Z_feat = linkage(condensed, method="ward")

    plt.figure(figsize=(50, 24), dpi=100)
    dendrogram(Z_feat, labels=significant, leaf_rotation=90, leaf_font_size=22)

    plt.title("Feature Dendrogram (Ward linkage, sign=1 features)", fontsize=38)
    plt.ylabel("Distance", fontsize=32)
    plt.tick_params(axis='y', labelsize=26)

    plt.subplots_adjust(left=0.10, bottom=0.35, right=0.97, top=0.90)

    plt.savefig(os.path.join(out_dir, "feature_dendrogram_sign_1.png"), dpi=150)
    plt.close()




# 5. PATIENT–PATIENT CO-CLUSTERING

'''
df_feat: DataFrame – patient-feature matrix containing only significant features
significant: list of str – names of significant features (sign=1)
out_dir: str – directory where co-clustering outputs will be saved (from config.toml)

This function:
    Performs patient–patient co-clustering across feature families using Ward linkage.
    Saves:
        - co-clustering CSV
        - co-clustering heatmap
'''
def co_clustering(df_feat, significant, out_dir):

    patients = df_feat["Patient"].values
    n_patients = len(patients)

    base_to_cols = {}
    for col in significant:
        base = col.split("_")[0]
        base_to_cols.setdefault(base, []).append(col)

    PER_FAMILY_N_CLUSTERS = 2
    min_cols_per_family = 1

    co_counts = np.zeros((n_patients, n_patients), dtype=float)
    family_counts = 0

    print("Starting per-feature-family patient clustering (significant features only)...")

    for base, cols in sorted(base_to_cols.items()):

        if len(cols) < min_cols_per_family:
            continue

        subX = df_feat[cols].copy()

        var_f = subX.var(axis=0)
        nonzero_cols_f = var_f[var_f > 0].index.tolist()
        if len(nonzero_cols_f) == 0:
            continue

        subX = subX[nonzero_cols_f]

        scaler_f = StandardScaler()
        subX_scaled = scaler_f.fit_transform(subX)

        Z_fam = linkage(subX_scaled, method="ward")
        labels_fam = fcluster(Z_fam, PER_FAMILY_N_CLUSTERS, criterion="maxclust")

        family_counts += 1

        for i in range(n_patients):
            for j in range(i + 1, n_patients):
                if labels_fam[i] == labels_fam[j]:
                    co_counts[i, j] += 1
                    co_counts[j, i] += 1

    print(f"Used {family_counts} feature families for patient co-clustering.")

    if family_counts == 0:
        print("No feature families available for co-clustering.")
        return

    co_fraction = co_counts / float(family_counts)

    co_df = pd.DataFrame(co_fraction, index=patients, columns=patients)
    co_df.to_csv(os.path.join(out_dir, "patient_patient_co_clustering_fraction_sign_1.csv"))

    plt.figure(figsize=(10, 8))
    plt.imshow(co_fraction, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Fraction of families co-clustered")
    plt.xticks(range(n_patients), patients, rotation=90)
    plt.yticks(range(n_patients), patients)
    plt.title("Patient–Patient Co-Clustering Fraction (Ward, significant features)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "patient_patient_co_clustering_fraction_sign_1.png"), dpi=300)
    plt.close()
