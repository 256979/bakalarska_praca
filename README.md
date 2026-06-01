# Automated Thrombus Analysis Pipeline — Bachelor’s Thesis Documentation

This repository contains a fully automated pipeline for thrombus analysis from mCTA imaging.
The system performs feature extraction, histogram analysis, per‑feature clustering, significance testing,
patient‑level clustering, and thrombus slice export. This document describes the implemented software pipeline,
its algorithmic structure, input/output formats, properties, limitations, and potential applications.

# 0. Running the Pipeline

The pipeline was implemented and tested on Linux.

Run the pipeline using:
    python main.py

The script loads configuration parameters from config.toml using the tomli library. It reads ROOT and OUT_CSV,
loads all export directory names, the p‑value threshold, and the silhouette threshold, executes all processing steps,
and writes outputs into the configured folders.

# Configuration File (config.toml)

All adjustable parameters are defined here.

[paths]
ROOT = "/mnt/md0/data/thrombus_dataset"
OUT_CSV = "/mnt/md0/results/thrombus_analysis"

If ROOT does not exist, the pipeline prints an error and stops.
If OUT_CSV does not exist, it is created automatically.
Both must be absolute paths.

Command‑line override:
    python main.py --root /path/to/data --out /path/to/output

If no overrides are provided, values from config.toml are used.

[export]
histogram_dir = "histograms"
plots_dir = "plots"
significant_feature_dir = "significant_feature_analysis"
slice_export_dir = "thrombus_slices"

These folders are created automatically if missing.

[statistics]
p_value_threshold = 0.05
silhouette_threshold = 0.50

These thresholds are used in cluster_plots_stats.py.

# 0.1 Dataset Structure

    bakalarska_praca/
    │
    ├── main.py
    ├── config.toml
    ├── pyproject.toml
    ├── README.md
    ├
    │
    └── sec/
        └── thrombus_pipeline/
            ├── cluster_plots_stats.py
            ├── feature_and_patient_level_overview.py
            ├── histogram_function.py
            ├── loader_functions.py
            └── thrombus_viewing.py

# 1. Dataset Requirements

## 1.1 Dataset Structure

    ROOT/
        Patient_001/
            Native/
                0001.dcm
                0002.dcm
                ...
            P1.nii.gz
            P2.nii.gz
            P3.nii.gz
            thrombus_mask.nii.gz
    
        Patient_002/
            Native/
            P1.nii.gz
            P2.nii.gz
            P3.nii.gz
            thrombus_mask.nii.gz

## 1.2 Required Modalities

Modality | Format | Description
Native | DICOM | Non‑contrast CT
P1 | NIfTI | First contrast phase
P2 | NIfTI | Second contrast phase
P3 | NIfTI | Third contrast phase
thrombus_mask | NIfTI | Binary mask aligned to all phases

All NIfTI volumes must share identical geometry.
The thrombus mask must align with all phases.
Missing files cause the patient to be skipped.

## 1.3 Naming Requirements

These naming rules are mandatory. Misnamed files or folders cause skipping or failure.

### 1.3.1 Patient Folder Names

Folder name must contain only digits (folder_name.isdigit()).

Valid: 15, 002, 73
Invalid: Patient_15, P15, 15a, patient15

### 1.3.2 Native CT Folder Name

Must be named exactly "nativ" or "native" (case‑insensitive).

Valid: nativ, NATIV, Nativ, Native
Invalid: DICOM, CT

Must contain a single DICOM series.

### 1.3.3 Segmentation File Names

Must contain "seg" in the filename.

Valid: thrombus_seg.nii.gz, segmentation.nii.gz, 15_seg_mask.nii
Invalid: mask.nii.gz, thrombus_mask.nii.gz, label_mask.nii.gz

Must be a NIfTI file and non‑empty.

### 1.3.4 Phase File Naming Requirements

Phase files must:
- be NIfTI (.nii or .nii.gz)
- not contain "seg", "mask", "label"
- contain an integer indicating the phase number

Prefix removal rule:
18_phase1_registered.nii.gz → prefix removed → phase1_registered.nii.gz → phase = 1

Exactly one file must exist for each phase: P1, P2, P3.

### 1.3.5 Orientation Requirements

All NIfTI files must be in RAS+ orientation or convertible to it.
Orientation codes must match across CT, segmentation, P1, P2, P3.
If not, the patient is skipped.

# 2. Software

Python version: 3.10.12

External packages:

numpy — numerical operations
pandas — CSV I/O
matplotlib — plotting
scipy — clustering, statistics
scikit‑learn — standardization, silhouette score
nibabel — loading NIfTI
pydicom — reading DICOM
SimpleITK — robust DICOM loading
tomli — reading TOML (Python ≤3.10)



Install dependencies:
    pip install -r requirements.txt

Or:
    pip install numpy pandas matplotlib scipy scikit-learn nibabel pydicom SimpleITK tomli

# 3. Pipeline Overview

The pipeline consists of five modules located in sec/thrombus_pipeline, executed by main.py:

1. feature_extraction.py — feature extraction
2. histogram_function.py — histogram generation
3. cluster_plots_stats.py — feature clustering + statistics
4. feature_and_patient_overview.py — significant feature analysis
5. thrombus_viewing.py — thrombus slice export

main.py runs the entire pipeline and produces all outputs.

# 4. Module Documentation

Briefly describes the function of each module - more detailed description is located in the code

## 4.1 feature_extraction_new.py

Extracts intensity‑based features from thrombus regions across all CT phases.

Returns a DataFrame containing:
- patient ID
- mean, SD, min, max
- median
- percentiles (P10, P25, P75, P90)
- skewness, kurtosis
- energy, entropy
- range, IQR
- phase‑to‑phase deltas
- native‑to‑phase deltas

## 4.2 histogram_function.py

Generates histograms for each patient and phase.

Output:
histograms/<patient_id>_histograms.png

## 4.3 cluster_plots_stats.py

Per‑feature hierarchical clustering, silhouette scoring, normality testing, statistical testing, and significance assignment.

Outputs:
- cluster_stats_k2.csv
- dendrograms
- scatterplots

## 4.4 feature_and_patient_overview.py

Performs:
- selection of significant features
- patient‑level clustering
- feature correlation analysis
- feature dendrogram
- patient co‑clustering

## 4.5 thrombus_viewing.py

Exports representative thrombus slices for qualitative inspection.

Output:
thrombus_slices/<patient_id>_slice.png

# 5. Outputs

All outputs are stored in OUT_CSV.

## 5.1 Feature Extraction
features_all_patients.csv

## 5.2 Histograms
histograms/<patient_id>_histograms.png

## 5.3 Per‑Feature Clustering
cluster_stats_k2.csv, dendrograms, scatterplots

## 5.4 Patient‑Level Clustering
patient_clusters_ward_sign_1.csv
dendrogram_ward_sign_1.png
correlation matrix + heatmap
feature dendrogram

## 5.5 Co‑Clustering
co‑clustering matrix
co‑clustering heatmap

## 5.6 Thrombus Slice Export
thrombus_slices/<patient_id>_slice.png

# 6. Limitations

- Strict dataset naming and structure required
- Fully automated — no manual overrides
- Runtime increases with dataset size
- Minor ITK warnings expected for clinical CT data

# 7. Summary

The pipeline performs:
- intensity‑based feature extraction
- histogram generation
- hierarchical clustering and significance testing
- patient‑level clustering
- feature correlation and dendrogram analysis
- patient co‑clustering
- thrombus slice export

# 8. Possible Applications

The pipeline can support studies relating thrombus imaging characteristics to:
- histology
- treatment response
- recanalization success
- functional outcomes
- biological heterogeneity

# 9. Differences Between Thesis Results and Pipeline Output

Development‑phase results may differ due to occasional manual adjustments.
The final pipeline is fully automated and does not allow manual intervention,
so outputs may not match development‑phase results exactly.

# Note on ITK Warning

A warning about non‑uniform sampling is expected for clinical CT datasets and does not indicate a problem.
It results from limited floating‑point precision in DICOM metadata.
