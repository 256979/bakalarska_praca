# Automated Thrombus Analysis Pipeline — Bachelor’s Thesis Documentation

Automated pipeline for thrombus analysis from multiphase CT angiography (mCTA).
The system performs feature extraction, histogram analysis, per‑feature clustering,
statistical testing, patient‑level clustering, and thrombus slice export.

The pipeline was fully implemented and tested on Linux.

This document describes the software environment, installation steps, dataset requirements,
configuration, and module‑level functionality. More detailed descriptions of inputs and outputs
are included directly inside the code of each module.

# 1. Environment

Python version:
    Python 3.10.12

Required packages (requirements.txt is provided):

requirements = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "matplotlib": "3.10.8",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",  
    "nibabel": "5.3.3",
    "pydicom": "3.0.1",
    "SimpleITK": "2.5.3",
    "tomli": "2.4.1",
}
# 2. Project Structure
    bakalarska_praca/
    │
    ├── main.py
    ├── config.toml
    ├── pyproject.toml
    ├── README.md
    │
    └── sec/
        └── thrombus_pipeline/
            ├── feature_extraction.py
            ├── cluster_plots_stats.py
            ├── feature_and_patient_level_overview.py
            ├── histogram_function.py
            ├── loader_functions.py
            └── thrombus_viewing.py


# 3. Step‑by‑Step Installation and Running

## 3.1 Install Python
Install Python 3.10.x from:
    https://www.python.org/downloads/

Verify installation:
    python3 --version

## 3.2 Download the repository
Either:
    
    git clone https://github.com/256979/bakalarska_praca

    cd bakalarska_praca

Or download ZIP → extract → open terminal inside the folder.

The folder must contain the structure shown above.

## 3.3 Create and activate a virtual environment (recommended)

Create:
   
     python3 -m venv .venv

Activate:
Linux/macOS:

        source .venv/bin/activate

Windows PowerShell:

        .venv\Scripts\Activate.ps1

Verify activation:
    (.venv) appears in the terminal prompt.

## 3.4 Install dependencies

Install:
    
    pip install -r requirements.txt

Verify:
    
    pip list

## 3.5 Configure paths in config.toml

[paths]
ROOT = "/absolute/path/to/dataset"
OUT_CSV = "/absolute/path/to/output"

[export]
histogram_dir = "histograms"
plots_dir = "plots"
significant_feature_dir = "significant_feature_analysis"
slice_export_dir = "thrombus_slices"

[statistics]
p_value_threshold = 0.05
silhouette_threshold = 0.50

Note:
    Paths can also be set through command line when running main.py:
        python3 main.py --root /path/to/data --out /path/to/output

## 3.6 Dataset structure (visualisation preserved)

    ROOT/
        001/
            native/
                0001.dcm
                0002.dcm
                ...
            P1.nii.gz
            P2.nii.gz
            P3.nii.gz
            thrombus_seg.nii.gz

        002/
            native/
            P1.nii.gz
            P2.nii.gz
            P3.nii.gz
            thrombus_seg.nii.gz

## 3.7 Dataset rules (must be satisfied)

    - Patient folder name must contain digits only.
    - Native CT folder must be named “native” or “nativ”.
    - Segmentation file must contain “seg” in the filename.
    - Phase files must be NIfTI and contain a phase number.
    - All NIfTI files must share identical geometry and RAS+ orientation.

If these conditions are not met, the patient is skipped.

## 3.8 Run the pipeline

Standard run (uses config.toml):
    python main.py

Run with command‑line overrides:
    python3 main.py --root /path/to/dataset --out /path/to/output

The script will:
    - load configuration
    - scan all patients in ROOT
    - run feature extraction, histograms, clustering, and slice export
    - write outputs into OUT_CSV and its subfolders

![]("C:\Users\mataf\OneDrive\Documents\pipeline.pdf")

## 3.9 Verifying successful execution

The pipeline ran successfully **provided that the dataset contained files meeting all required criteria**
and the following folders contain non‑empty outputs:

    OUT_CSV/features_all_patients.csv
    OUT_CSV/histograms/
    OUT_CSV/plots/
    OUT_CSV/significant_feature_analysis/
    OUT_CSV/thrombus_slices/

# 4. Pipeline Modules

The pipeline consists of modules executed by main.py.
Each module produces its own outputs. A more detailed description of inputs,
outputs, and internal processing is included directly in the code of each module.

# 4.1 loader_functions.py

Purpose:
    Provides all low‑level data loading utilities for the pipeline, including:
        - DICOM reconstruction (native CT)
        - NIfTI loading and RAS+ reorientation
        - segmentation discovery and validation
        - phase discovery (P1, P2, P3) and consistency checks
        - full patient‑level data assembly

Main operations:
- load_dicom_series():
    * reads a full DICOM series using SimpleITK
    * reconstructs a 3D CT volume
    * converts LPS → RAS orientation
    * reorients into canonical RAS+
    * returns voxel array + orientation codes

- load_nifti():
    * loads .nii / .nii.gz files
    * checks orientation
    * reorients to RAS+ if needed
    * returns voxel array + axis codes

- extract_phase_number():
    * parses filenames
    * ignores segmentation files
    * extracts numeric phase identifiers

- find_patients():
    * scans dataset root
    * selects folders containing digits only
    * returns sorted patient list

- find_dicom():
    * recursively searches for “native” / “nativ” folder
    * returns path to CT DICOM directory

- find_and_load_segmentation():
    * finds segmentation file (*seg*.nii*)
    * loads and binarizes mask
    * validates non‑empty segmentation

- find_and_load_phases():
    * finds all NIfTI phase files
    * extracts phase numbers
    * loads P1, P2, P3
    * checks shape + orientation consistency with segmentation

- load_patient_data():
    * loads CT, segmentation, and all phases
    * validates geometry and orientation
    * extracts thrombus voxels for native + all phases
    * returns a structured dictionary with:
          patient_id
          masked voxel intensities
          full CT volumes
          segmentation mask

Output:
    A dictionary containing:
        - full CT volumes (native, P1, P2, P3)
        - segmentation mask
        - voxel‑level thrombus intensities for all phases
        - patient ID


# 4.2 feature_extraction.py

Purpose:
    Extracts intensity‑based features from thrombus regions across all CT phases.

Main operations:
    - loads thrombus mask and CT phases
    - computes statistical descriptors (mean, SD, min, max, percentiles)
    - computes skewness, kurtosis, entropy, energy
    - computes phase‑to‑phase and native‑to‑phase deltas

Output:
    features_all_patients.csv


# 4.3 histogram_function.py

Purpose:
    Generates histograms for each patient and phase.

Main operations:
    - extracts voxel intensities from thrombus mask
    - computes histogram distributions
    - plots histograms for all phases

Output:
    histograms/<patient_id>_histograms.png


# 4.4 cluster_plots_stats.py

Purpose:
    Performs per‑feature hierarchical clustering and statistical testing.

Main operations:
    - standardizes features
    - hierarchical clustering (Ward)
    - silhouette scoring
    - normality testing
    - Mann‑Whitney U tests
    - significance assignment

Output:
    cluster_stats_k2.csv
    dendrograms and scatterplots


# 4.5 feature_and_patient_level_overview.py

Purpose:
    Performs patient‑level clustering and feature significance analysis.

Main operations:
    - selects significant features
    - computes correlation matrices
    - generates dendrograms
    - performs patient co‑clustering

Output:
    patient_clusters_ward_sign_1.csv
    correlation heatmaps and dendrograms


# 4.6 thrombus_viewing.py

Purpose:
    Exports representative thrombus slices for qualitative inspection.

Main operations:
    - identifies slices containing thrombus
    - centers thrombus in a fixed canvas
    - draws pixel‑accurate thrombus outline
    - exports slices for all phases and patients

Output:
    thrombus_slices/<patient_id>/<phase>/slice_XXX.png


# 5. Limitations

- Strict dataset structure required.
- Fully automated pipeline; no manual overrides.
- Runtime increases with dataset size.

- ITK Warning:
    During DICOM loading, SimpleITK may issue a warning such as:
        “Non‑uniform sampling or missing slices detected”
    This warning is expected for clinical CT datasets and does not indicate missing slices.
    It results from limited floating‑point precision in DICOM metadata.
    The reconstructed volume is valid and the warning does not affect feature extraction,
    segmentation alignment, or any downstream analysis.

# 6. Summary

The pipeline provides:
    - automated feature extraction
    - histogram generation
    - hierarchical clustering and statistical testing
    - patient‑level clustering and correlation analysis
    - export of representative thrombus slices

It is suitable for:
    - studying thrombus heterogeneity
    - exploring quantitative imaging biomarkers
    - analysing multiphase CT thrombus characteristics
    - generating reproducible imaging‑based datasets for research
