import os
import pandas as pd
import tomli

from sec.thrombus_pipeline.thrombus_viewing import export_thrombus_slices
from sec.thrombus_pipeline.loader_functions import load_patient_data, find_patients
from sec.thrombus_pipeline.feature_extraction import extract_features_for_all_patients
from sec.thrombus_pipeline.histogram_function import plot_histograms

from sec.thrombus_pipeline.cluster_plots_stats import (
    load_feature_csv,
    process_all_features
)

from sec.thrombus_pipeline.feature_and_patient_level_overview import (
    load_data,
    select_significant_features,
    cluster_patients,
    analyze_features,
    co_clustering
)

def main(root=None, out_csv=None):

    # Load configuration
    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    # ROOT and OUT_CSV as main parameters
    ROOT = root if root is not None else config["paths"]["ROOT"]
    OUT_CSV = out_csv if out_csv is not None else config["paths"]["OUT_CSV"]

    print("ROOT USED:", ROOT)
    print("EXAMPLE CONTENTS:", os.listdir(ROOT)[:10])

    os.makedirs(OUT_CSV, exist_ok=True)

    # 1. FEATURE EXTRACTION
    df = extract_features_for_all_patients(ROOT)

    feature_csv_path = os.path.join(OUT_CSV, "features_all_patients.csv")
    df.to_csv(feature_csv_path, index=False)
    print(f"Saved feature table: {feature_csv_path}")

    # 2. HISTOGRAM GENERATION
    histogram_dir = os.path.join(OUT_CSV, config["export"]["histogram_dir"])
    os.makedirs(histogram_dir, exist_ok=True)

    print("\nGenerating histograms...")

    patient_dirs = find_patients(ROOT)

    for patient_dir in patient_dirs:
        data = load_patient_data(patient_dir)
        if data is None:
            continue

        pid = data["patient_id"]

        plot_histograms(
            data["vox_Native"],
            data["vox_P1"],
            data["vox_P2"],
            data["vox_P3"],
            pid,
            histogram_dir
        )

        print(f"Histogram saved for patient {pid}")

    # 3. FEATURE-LEVEL CLUSTERING + STATISTICS
    print("\nRunning feature-level clustering/statistics pipeline...")

    df_loaded, patient_ids, features = load_feature_csv(feature_csv_path)

    plot_dir = os.path.join(OUT_CSV, config["export"]["plots_dir"])
    os.makedirs(plot_dir, exist_ok=True)

    stats_rows = process_all_features(
        df_loaded,
        patient_ids,
        features,
        plot_dir,
        config,
        k=2
    )

    stats_path = os.path.join(OUT_CSV, "cluster_stats_k2.csv")
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    print(f"\nFeature-level clustering complete. Results saved to: {stats_path}")

    # 4. SIGNIFICANT-FEATURE PATIENT-LEVEL ANALYSIS
    print("\nRunning significant-feature patient-level analysis...")

    df_feat, df_sign = load_data(feature_csv_path, stats_path)
    df_feat_sig, available = select_significant_features(df_feat, df_sign)

    sig_dir = os.path.join(OUT_CSV, config["export"]["significant_feature_dir"])
    os.makedirs(sig_dir, exist_ok=True)

    cluster_patients(df_feat_sig, available, sig_dir)
    analyze_features(df_feat_sig, available, sig_dir)
    co_clustering(df_feat_sig, available, sig_dir)

    print("\nSignificant-feature patient-level analysis complete.")

    # 5. THROMBUS SLICE EXPORT
    print("\nExporting thrombus slices...")

    slice_dir = os.path.join(OUT_CSV, config["export"]["slice_export_dir"])
    os.makedirs(slice_dir, exist_ok=True)

    export_thrombus_slices(ROOT, slice_dir)

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()



