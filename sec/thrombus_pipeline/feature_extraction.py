import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
#if does not work - try deleting sec.thrombus_pipeline or adding bakalarska_praca. before sec - see README
from sec.thrombus_pipeline.loader_functions import load_patient_data, find_patients



#ENTROPY FUNCTION

'''
vox_int: 1D NumPy array – voxel intensities inside the thrombus mask
bins: int – number of histogram bins (default: 256)
output: float – histogram‑based Shannon entropy

This function:
    Computes a histogram of voxel intensities using a fixed number of bins
    Converts histogram counts into probabilities
    Removes zero‑probability bins to avoid log(0)
    Computes Shannon entropy as −Σ p·log₂(p)
    Returns the entropy value as a float
'''
def _entropy(vox_int, bins=256):
    h, _ = np.histogram(vox_int, bins=bins)
    p = h.astype(np.float64)
    s = p.sum()
    if s == 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

#UNIFORMITY FUNCTION
'''
vox_int: 1D NumPy array – voxel intensities inside the thrombus mask
bins: int – number of histogram bins (default: 256)
output: float – histogram‑based uniformity (energy)

This function:
    Computes a histogram of voxel intensities using a fixed number of bins
    Converts histogram counts into probabilities
    Computes uniformity as Σ p²
    Returns the uniformity value as a float
'''

def _uniformity(vox_int, bins=256):
    h, _ = np.histogram(vox_int, bins=bins)
    p = h.astype(np.float64)
    s = p.sum()
    if s == 0:
        return 0.0
    p /= s
    return np.sum(p ** 2)

#FEATURE EXTRACTION
'''
ROOT: str – path to the dataset root directory
output: pandas.DataFrame – one row per patient containing all extracted features

This function:
    Finds all patient directories by calling find_patients()
    Iterates through each patient folder
    Loads CT, segmentation, and phase volumes by calling load_patient_data()
    Extracts thrombus voxels for Native, P1, P2, P3
    Computes delta arrays between phases
    Computes statistical features :Mean, Median, Min, Max, Range, IQR, SD, Variance
        Skewness, Kurtosis, Entropy , Uniformity 
    Computes MIP‑based features across phases
    Computes adjacent‑phase delta summaries (max/min)
    Computes native‑phase delta summaries (max/min)
    Stores all features in a dictionary
    Appends each dictionary as a row in a list
    Returns a DataFrame containing all patients and all features
'''
def extract_features_for_all_patients(ROOT):
    # FIND PATIENT DIRECTORIES

    patient_dirs = find_patients(ROOT)

    rows = []

    # MAIN LOOP

    for patient_dir in patient_dirs:

        patient_id = os.path.basename(patient_dir)

        # LOAD ALL PATIENT DATA

        data = load_patient_data(patient_dir)
        if data is None:
            continue

        vox_Native = data["vox_Native"]
        vox_P1 = data["vox_P1"]
        vox_P2 = data["vox_P2"]
        vox_P3 = data["vox_P3"]

        # 3D tMIP ACROSS PHASES (P1, P2, P3)



        data_P1 = data["full_P1"]
        data_P2 = data["full_P2"]
        data_P3 = data["full_P3"]
        seg = data["seg_mask"]


        Proj_MIP_3D = np.maximum.reduce([data_P1, data_P2, data_P3])

        # thrombus voxels
        vox_MIP = Proj_MIP_3D[seg]

        # MIP STATISTICS
        Proj_MIP_Mean = float(np.mean(vox_MIP))
        Proj_MIP_Std = float(np.std(vox_MIP))
        Proj_MIP_Skew = float(skew(vox_MIP))
        Proj_MIP_Kurt = float(kurtosis(vox_MIP))
        Proj_MIP_IQR = float(np.percentile(vox_MIP, 75) - np.percentile(vox_MIP, 25))

        # DELTA ARRAYS

        vox_Delta_N_P1 = vox_P1 - vox_Native
        vox_Delta_N_P2 = vox_P2 - vox_Native
        vox_Delta_N_P3 = vox_P3 - vox_Native
        vox_Delta_P1_P2 = vox_P2 - vox_P1
        vox_Delta_P2_P3 = vox_P3 - vox_P2
        vox_Delta_P1_P3 = vox_P3 - vox_P1

        # MEAN

        Mean_Native = vox_Native.mean()
        Mean_P1 = vox_P1.mean()
        Mean_P2 = vox_P2.mean()
        Mean_P3 = vox_P3.mean()

        Mean_Delta_N_P1 = Mean_P1 - Mean_Native
        Mean_Delta_N_P2 = Mean_P2 - Mean_Native
        Mean_Delta_N_P3 = Mean_P3 - Mean_Native
        Mean_Delta_P1_P2 = Mean_P2 - Mean_P1
        Mean_Delta_P2_P3 = Mean_P3 - Mean_P2
        Mean_Delta_P1_P3 = Mean_P3 - Mean_P1

        # MEDIAN

        Median_Native = np.median(vox_Native)
        Median_P1 = np.median(vox_P1)
        Median_P2 = np.median(vox_P2)
        Median_P3 = np.median(vox_P3)

        Median_Delta_N_P1 = Median_P1 - Median_Native
        Median_Delta_N_P2 = Median_P2 - Median_Native
        Median_Delta_N_P3 = Median_P3 - Median_Native
        Median_Delta_P1_P2 = Median_P2 - Median_P1
        Median_Delta_P2_P3 = Median_P3 - Median_P2
        Median_Delta_P1_P3 = Median_P3 - Median_P1


        # MINIMUM

        Min_Native = np.min(vox_Native)
        Min_P1 = np.min(vox_P1)
        Min_P2 = np.min(vox_P2)
        Min_P3 = np.min(vox_P3)


        # MAXIMUM

        Max_Native = np.max(vox_Native)
        Max_P1 = np.max(vox_P1)
        Max_P2 = np.max(vox_P2)
        Max_P3 = np.max(vox_P3)

        # SKEWNESS

        Skew_Native = skew(vox_Native)
        Skew_P1 = skew(vox_P1)
        Skew_P2 = skew(vox_P2)
        Skew_P3 = skew(vox_P3)

        Skew_Delta_N_P1 = skew(vox_Delta_N_P1)
        Skew_Delta_N_P2 = skew(vox_Delta_N_P2)
        Skew_Delta_N_P3 = skew(vox_Delta_N_P3)
        Skew_Delta_P1_P2 = skew(vox_Delta_P1_P2)
        Skew_Delta_P2_P3 = skew(vox_Delta_P2_P3)
        Skew_Delta_P1_P3 = skew(vox_Delta_P1_P3)


        # KURTOSIS

        Kurt_Native = kurtosis(vox_Native)
        Kurt_P1 = kurtosis(vox_P1)
        Kurt_P2 = kurtosis(vox_P2)
        Kurt_P3 = kurtosis(vox_P3)

        Kurt_Delta_N_P1 = kurtosis(vox_Delta_N_P1)
        Kurt_Delta_N_P2 = kurtosis(vox_Delta_N_P2)
        Kurt_Delta_N_P3 = kurtosis(vox_Delta_N_P3)
        Kurt_Delta_P1_P2 = kurtosis(vox_Delta_P1_P2)
        Kurt_Delta_P2_P3 = kurtosis(vox_Delta_P2_P3)
        Kurt_Delta_P1_P3 = kurtosis(vox_Delta_P1_P3)


        # ENTROPY

        Entropy_Native = _entropy(vox_Native)
        Entropy_P1 = _entropy(vox_P1)
        Entropy_P2 = _entropy(vox_P2)
        Entropy_P3 = _entropy(vox_P3)

        Entropy_Delta_N_P1 = _entropy(vox_Delta_N_P1)
        Entropy_Delta_N_P2 = _entropy(vox_Delta_N_P2)
        Entropy_Delta_N_P3 = _entropy(vox_Delta_N_P3)
        Entropy_Delta_P1_P2 = _entropy(vox_Delta_P1_P2)
        Entropy_Delta_P2_P3 = _entropy(vox_Delta_P2_P3)
        Entropy_Delta_P1_P3 = _entropy(vox_Delta_P1_P3)


        # UNIFORMITY

        Uniformity_Native = _uniformity(vox_Native)
        Uniformity_P1 = _uniformity(vox_P1)
        Uniformity_P2 = _uniformity(vox_P2)
        Uniformity_P3 = _uniformity(vox_P3)

        Uniformity_Delta_N_P1 = _uniformity(vox_Delta_N_P1)
        Uniformity_Delta_N_P2 = _uniformity(vox_Delta_N_P2)
        Uniformity_Delta_N_P3 = _uniformity(vox_Delta_N_P3)
        Uniformity_Delta_P1_P2 = _uniformity(vox_Delta_P1_P2)
        Uniformity_Delta_P2_P3 = _uniformity(vox_Delta_P2_P3)
        Uniformity_Delta_P1_P3 = _uniformity(vox_Delta_P1_P3)


        # RANGE

        Range_Native = Max_Native - Min_Native
        Range_P1 = Max_P1 - Min_P1
        Range_P2 = Max_P2 - Min_P2
        Range_P3 = Max_P3 - Min_P3

        Range_Delta_N_P1 = np.max(vox_Delta_N_P1) - np.min(vox_Delta_N_P1)
        Range_Delta_N_P2 = np.max(vox_Delta_N_P2) - np.min(vox_Delta_N_P2)
        Range_Delta_N_P3 = np.max(vox_Delta_N_P3) - np.min(vox_Delta_N_P3)
        Range_Delta_P1_P2 = np.max(vox_Delta_P1_P2) - np.min(vox_Delta_P1_P2)
        Range_Delta_P2_P3 = np.max(vox_Delta_P2_P3) - np.min(vox_Delta_P2_P3)
        Range_Delta_P1_P3 = np.max(vox_Delta_P1_P3) - np.min(vox_Delta_P1_P3)


        # INTERQUARTILE RANGE (IQR)

        IQR_Native = np.percentile(vox_Native, 75) - np.percentile(vox_Native, 25)
        IQR_P1 = np.percentile(vox_P1, 75) - np.percentile(vox_P1, 25)
        IQR_P2 = np.percentile(vox_P2, 75) - np.percentile(vox_P2, 25)
        IQR_P3 = np.percentile(vox_P3, 75) - np.percentile(vox_P3, 25)

        IQR_Delta_N_P1 = np.percentile(vox_Delta_N_P1, 75) - np.percentile(vox_Delta_N_P1, 25)
        IQR_Delta_N_P2 = np.percentile(vox_Delta_N_P2, 75) - np.percentile(vox_Delta_N_P2, 25)
        IQR_Delta_N_P3 = np.percentile(vox_Delta_N_P3, 75) - np.percentile(vox_Delta_N_P3, 25)
        IQR_Delta_P1_P2 = np.percentile(vox_Delta_P1_P2, 75) - np.percentile(vox_Delta_P1_P2, 25)
        IQR_Delta_P2_P3 = np.percentile(vox_Delta_P2_P3, 75) - np.percentile(vox_Delta_P2_P3, 25)
        IQR_Delta_P1_P3 = np.percentile(vox_Delta_P1_P3, 75) - np.percentile(vox_Delta_P1_P3, 25)


        # STANDARD DEVIATION

        SD_Native = vox_Native.std(ddof=1)
        SD_P1 = vox_P1.std(ddof=1)
        SD_P2 = vox_P2.std(ddof=1)
        SD_P3 = vox_P3.std(ddof=1)

        SD_Delta_N_P1 = SD_P1 - SD_Native
        SD_Delta_N_P2 = SD_P2 - SD_Native
        SD_Delta_N_P3 = SD_P3 - SD_Native

        SD_Delta_P1_P2 = SD_P2 - SD_P1
        SD_Delta_P2_P3 = SD_P3 - SD_P2
        SD_Delta_P1_P3 = SD_P3 - SD_P1


        # VARIANCE

        Var_Native = np.var(vox_Native, ddof=1)
        Var_P1 = np.var(vox_P1, ddof=1)
        Var_P2 = np.var(vox_P2, ddof=1)
        Var_P3 = np.var(vox_P3, ddof=1)

        Var_Delta_N_P1 = np.var(vox_Delta_N_P1, ddof=1)
        Var_Delta_N_P2 = np.var(vox_Delta_N_P2, ddof=1)
        Var_Delta_N_P3 = np.var(vox_Delta_N_P3, ddof=1)
        Var_Delta_P1_P2 = np.var(vox_Delta_P1_P2, ddof=1)
        Var_Delta_P2_P3 = np.var(vox_Delta_P2_P3, ddof=1)
        Var_Delta_P1_P3 = np.var(vox_Delta_P1_P3, ddof=1)

        # ADJACENT DELTA

        # MEAN
        Max_Mean_Deltas_Adjacent = max(Mean_Delta_N_P1, Mean_Delta_P1_P2, Mean_Delta_P2_P3)
        Min_Mean_Deltas_Adjacent = min(Mean_Delta_N_P1, Mean_Delta_P1_P2, Mean_Delta_P2_P3)

        # MEDIAN
        Max_Median_Deltas_Adjacent = max(Median_Delta_N_P1, Median_Delta_P1_P2, Median_Delta_P2_P3)
        Min_Median_Deltas_Adjacent = min(Median_Delta_N_P1, Median_Delta_P1_P2, Median_Delta_P2_P3)

        # RANGE
        Max_Range_Deltas_Adjacent = max(Range_Delta_N_P1, Range_Delta_P1_P2, Range_Delta_P2_P3)
        Min_Range_Deltas_Adjacent = min(Range_Delta_N_P1, Range_Delta_P1_P2, Range_Delta_P2_P3)

        # IQR
        Max_IQR_Deltas_Adjacent = max(IQR_Delta_N_P1, IQR_Delta_P1_P2, IQR_Delta_P2_P3)
        Min_IQR_Deltas_Adjacent = min(IQR_Delta_N_P1, IQR_Delta_P1_P2, IQR_Delta_P2_P3)

        # SD
        Max_SD_Deltas_Adjacent = max(SD_Delta_N_P1, SD_Delta_P1_P2, SD_Delta_P2_P3)
        Min_SD_Deltas_Adjacent = min(SD_Delta_N_P1, SD_Delta_P1_P2, SD_Delta_P2_P3)

        # VAR
        Max_Var_Deltas_Adjacent = max(Var_Delta_N_P1, Var_Delta_P1_P2, Var_Delta_P2_P3)
        Min_Var_Deltas_Adjacent = min(Var_Delta_N_P1, Var_Delta_P1_P2, Var_Delta_P2_P3)

        # SKEW
        Max_Skew_Deltas_Adjacent = max(Skew_Delta_N_P1, Skew_Delta_P1_P2, Skew_Delta_P2_P3)
        Min_Skew_Deltas_Adjacent = min(Skew_Delta_N_P1, Skew_Delta_P1_P2, Skew_Delta_P2_P3)

        # KURT
        Max_Kurt_Deltas_Adjacent = max(Kurt_Delta_N_P1, Kurt_Delta_P1_P2, Kurt_Delta_P2_P3)
        Min_Kurt_Deltas_Adjacent = min(Kurt_Delta_N_P1, Kurt_Delta_P1_P2, Kurt_Delta_P2_P3)

        # ENTROPY
        Max_Entropy_Deltas_Adjacent = max(Entropy_Delta_N_P1, Entropy_Delta_P1_P2, Entropy_Delta_P2_P3)
        Min_Entropy_Deltas_Adjacent = min(Entropy_Delta_N_P1, Entropy_Delta_P1_P2, Entropy_Delta_P2_P3)

        # UNIFORMITY
        Max_Uniformity_Deltas_Adjacent = max(Uniformity_Delta_N_P1, Uniformity_Delta_P1_P2, Uniformity_Delta_P2_P3)
        Min_Uniformity_Deltas_Adjacent = min(Uniformity_Delta_N_P1, Uniformity_Delta_P1_P2, Uniformity_Delta_P2_P3)

        # NATIVE DELTA

        # MEAN
        Max_Mean_Deltas_Native = max(Mean_Delta_N_P1, Mean_Delta_N_P2, Mean_Delta_N_P3)
        Min_Mean_Deltas_Native = min(Mean_Delta_N_P1, Mean_Delta_N_P2, Mean_Delta_N_P3)

        # MEDIAN
        Max_Median_Deltas_Native = max(Median_Delta_N_P1, Median_Delta_N_P2, Median_Delta_N_P3)
        Min_Median_Deltas_Native = min(Median_Delta_N_P1, Median_Delta_N_P2, Median_Delta_N_P3)

        # RANGE
        Max_Range_Deltas_Native = max(Range_Delta_N_P1, Range_Delta_N_P2, Range_Delta_N_P3)
        Min_Range_Deltas_Native = min(Range_Delta_N_P1, Range_Delta_N_P2, Range_Delta_N_P3)

        # IQR
        Max_IQR_Deltas_Native = max(IQR_Delta_N_P1, IQR_Delta_N_P2, IQR_Delta_N_P3)
        Min_IQR_Deltas_Native = min(IQR_Delta_N_P1, IQR_Delta_N_P2, IQR_Delta_N_P3)

        # SD
        Max_SD_Deltas_Native = max(SD_Delta_N_P1, SD_Delta_N_P2, SD_Delta_N_P3)
        Min_SD_Deltas_Native = min(SD_Delta_N_P1, SD_Delta_N_P2, SD_Delta_N_P3)

        # VARIANCE
        Max_Var_Deltas_Native = max(Var_Delta_N_P1, Var_Delta_N_P2, Var_Delta_N_P3)
        Min_Var_Deltas_Native = min(Var_Delta_N_P1, Var_Delta_N_P2, Var_Delta_N_P3)

        # SKEWNESS
        Max_Skew_Deltas_Native = max(Skew_Delta_N_P1, Skew_Delta_N_P2, Skew_Delta_N_P3)
        Min_Skew_Deltas_Native = min(Skew_Delta_N_P1, Skew_Delta_N_P2, Skew_Delta_N_P3)

        # KURTOSIS
        Max_Kurt_Deltas_Native = max(Kurt_Delta_N_P1, Kurt_Delta_N_P2, Kurt_Delta_N_P3)
        Min_Kurt_Deltas_Native = min(Kurt_Delta_N_P1, Kurt_Delta_N_P2, Kurt_Delta_N_P3)

        # ENTROPY
        Max_Entropy_Deltas_Native = max(Entropy_Delta_N_P1, Entropy_Delta_N_P2, Entropy_Delta_N_P3)
        Min_Entropy_Deltas_Native = min(Entropy_Delta_N_P1, Entropy_Delta_N_P2, Entropy_Delta_N_P3)

        # UNIFORMITY
        Max_Uniformity_Deltas_Native = max(Uniformity_Delta_N_P1, Uniformity_Delta_N_P2, Uniformity_Delta_N_P3)
        Min_Uniformity_Deltas_Native = min(Uniformity_Delta_N_P1, Uniformity_Delta_N_P2, Uniformity_Delta_N_P3)

        # FEATURE DICTIONARY

        row = {
            "Patient": int(patient_id),

            # MEAN
            "Mean_Native": Mean_Native,
            "Mean_P1": Mean_P1,
            "Mean_P2": Mean_P2,
            "Mean_P3": Mean_P3,
            "Mean_Delta_N_P1": Mean_Delta_N_P1,
            "Mean_Delta_N_P2": Mean_Delta_N_P2,
            "Mean_Delta_N_P3": Mean_Delta_N_P3,
            "Mean_Delta_P1_P2": Mean_Delta_P1_P2,
            "Mean_Delta_P2_P3": Mean_Delta_P2_P3,
            "Mean_Delta_P1_P3": Mean_Delta_P1_P3,

            #  MEDIAN
            "Median_Native": Median_Native,
            "Median_P1": Median_P1,
            "Median_P2": Median_P2,
            "Median_P3": Median_P3,
            "Median_Delta_N_P1": Median_Delta_N_P1,
            "Median_Delta_N_P2": Median_Delta_N_P2,
            "Median_Delta_N_P3": Median_Delta_N_P3,
            "Median_Delta_P1_P2": Median_Delta_P1_P2,
            "Median_Delta_P2_P3": Median_Delta_P2_P3,
            "Median_Delta_P1_P3": Median_Delta_P1_P3,

            # RANGE
            "Range_Native": Range_Native,
            "Range_P1": Range_P1,
            "Range_P2": Range_P2,
            "Range_P3": Range_P3,
            "Range_Delta_N_P1": Range_Delta_N_P1,
            "Range_Delta_N_P2": Range_Delta_N_P2,
            "Range_Delta_N_P3": Range_Delta_N_P3,
            "Range_Delta_P1_P2": Range_Delta_P1_P2,
            "Range_Delta_P2_P3": Range_Delta_P2_P3,
            "Range_Delta_P1_P3": Range_Delta_P1_P3,

            # IQR
            "IQR_Native": IQR_Native,
            "IQR_P1": IQR_P1,
            "IQR_P2": IQR_P2,
            "IQR_P3": IQR_P3,
            "IQR_Delta_N_P1": IQR_Delta_N_P1,
            "IQR_Delta_N_P2": IQR_Delta_N_P2,
            "IQR_Delta_N_P3": IQR_Delta_N_P3,
            "IQR_Delta_P1_P2": IQR_Delta_P1_P2,
            "IQR_Delta_P2_P3": IQR_Delta_P2_P3,
            "IQR_Delta_P1_P3": IQR_Delta_P1_P3,

            #  SD
            "SD_Native": SD_Native,
            "SD_P1": SD_P1,
            "SD_P2": SD_P2,
            "SD_P3": SD_P3,
            "SD_Delta_N_P1": SD_Delta_N_P1,
            "SD_Delta_N_P2": SD_Delta_N_P2,
            "SD_Delta_N_P3": SD_Delta_N_P3,
            "SD_Delta_P1_P2": SD_Delta_P1_P2,
            "SD_Delta_P2_P3": SD_Delta_P2_P3,
            "SD_Delta_P1_P3": SD_Delta_P1_P3,

            # VARIANCE
            "Var_Native": Var_Native,
            "Var_P1": Var_P1,
            "Var_P2": Var_P2,
            "Var_P3": Var_P3,
            "Var_Delta_N_P1": Var_Delta_N_P1,
            "Var_Delta_N_P2": Var_Delta_N_P2,
            "Var_Delta_N_P3": Var_Delta_N_P3,
            "Var_Delta_P1_P2": Var_Delta_P1_P2,
            "Var_Delta_P2_P3": Var_Delta_P2_P3,
            "Var_Delta_P1_P3": Var_Delta_P1_P3,

            # SKEWNESS
            "Skew_Native": Skew_Native,
            "Skew_P1": Skew_P1,
            "Skew_P2": Skew_P2,
            "Skew_P3": Skew_P3,
            "Skew_Delta_N_P1": Skew_Delta_N_P1,
            "Skew_Delta_N_P2": Skew_Delta_N_P2,
            "Skew_Delta_N_P3": Skew_Delta_N_P3,
            "Skew_Delta_P1_P2": Skew_Delta_P1_P2,
            "Skew_Delta_P2_P3": Skew_Delta_P2_P3,
            "Skew_Delta_P1_P3": Skew_Delta_P1_P3,

            # KURTOSIS
            "Kurt_Native": Kurt_Native,
            "Kurt_P1": Kurt_P1,
            "Kurt_P2": Kurt_P2,
            "Kurt_P3": Kurt_P3,
            "Kurt_Delta_N_P1": Kurt_Delta_N_P1,
            "Kurt_Delta_N_P2": Kurt_Delta_N_P2,
            "Kurt_Delta_N_P3": Kurt_Delta_N_P3,
            "Kurt_Delta_P1_P2": Kurt_Delta_P1_P2,
            "Kurt_Delta_P2_P3": Kurt_Delta_P2_P3,
            "Kurt_Delta_P1_P3": Kurt_Delta_P1_P3,

            #  ENTROPY
            "Entropy_Native": Entropy_Native,
            "Entropy_P1": Entropy_P1,
            "Entropy_P2": Entropy_P2,
            "Entropy_P3": Entropy_P3,
            "Entropy_Delta_N_P1": Entropy_Delta_N_P1,
            "Entropy_Delta_N_P2": Entropy_Delta_N_P2,
            "Entropy_Delta_N_P3": Entropy_Delta_N_P3,
            "Entropy_Delta_P1_P2": Entropy_Delta_P1_P2,
            "Entropy_Delta_P2_P3": Entropy_Delta_P2_P3,
            "Entropy_Delta_P1_P3": Entropy_Delta_P1_P3,

            # UNIFORMITY
            "Uniformity_Native": Uniformity_Native,
            "Uniformity_P1": Uniformity_P1,
            "Uniformity_P2": Uniformity_P2,
            "Uniformity_P3": Uniformity_P3,
            "Uniformity_Delta_N_P1": Uniformity_Delta_N_P1,
            "Uniformity_Delta_N_P2": Uniformity_Delta_N_P2,
            "Uniformity_Delta_N_P3": Uniformity_Delta_N_P3,
            "Uniformity_Delta_P1_P2": Uniformity_Delta_P1_P2,
            "Uniformity_Delta_P2_P3": Uniformity_Delta_P2_P3,
            "Uniformity_Delta_P1_P3": Uniformity_Delta_P1_P3,

            # ADJACENT
            "Max_Mean_Deltas_Adjacent": Max_Mean_Deltas_Adjacent,
            "Min_Mean_Deltas_Adjacent": Min_Mean_Deltas_Adjacent,
            "Max_Median_Deltas_Adjacent": Max_Median_Deltas_Adjacent,
            "Min_Median_Deltas_Adjacent": Min_Median_Deltas_Adjacent,

            "Max_Range_Deltas_Adjacent": Max_Range_Deltas_Adjacent,
            "Min_Range_Deltas_Adjacent": Min_Range_Deltas_Adjacent,
            "Max_IQR_Deltas_Adjacent": Max_IQR_Deltas_Adjacent,
            "Min_IQR_Deltas_Adjacent": Min_IQR_Deltas_Adjacent,
            "Max_SD_Deltas_Adjacent": Max_SD_Deltas_Adjacent,
            "Min_SD_Deltas_Adjacent": Min_SD_Deltas_Adjacent,
            "Max_Var_Deltas_Adjacent": Max_Var_Deltas_Adjacent,
            "Min_Var_Deltas_Adjacent": Min_Var_Deltas_Adjacent,
            "Max_Skew_Deltas_Adjacent": Max_Skew_Deltas_Adjacent,
            "Min_Skew_Deltas_Adjacent": Min_Skew_Deltas_Adjacent,
            "Max_Kurt_Deltas_Adjacent": Max_Kurt_Deltas_Adjacent,
            "Min_Kurt_Deltas_Adjacent": Min_Kurt_Deltas_Adjacent,
            "Max_Entropy_Deltas_Adjacent": Max_Entropy_Deltas_Adjacent,
            "Min_Entropy_Deltas_Adjacent": Min_Entropy_Deltas_Adjacent,
            "Max_Uniformity_Deltas_Adjacent": Max_Uniformity_Deltas_Adjacent,
            "Min_Uniformity_Deltas_Adjacent": Min_Uniformity_Deltas_Adjacent,

            # MIP FEATURES
            "Proj_MIP_Mean": Proj_MIP_Mean,
            "Proj_MIP_Std": Proj_MIP_Std,
            "Proj_MIP_Skew": Proj_MIP_Skew,
            "Proj_MIP_Kurt": Proj_MIP_Kurt,
            "Proj_MIP_IQR": Proj_MIP_IQR,

            #  NATIVE DELTA
            "Max_Mean_Deltas_Native": Max_Mean_Deltas_Native,
            "Min_Mean_Deltas_Native": Min_Mean_Deltas_Native,

            "Max_Median_Deltas_Native": Max_Median_Deltas_Native,
            "Min_Median_Deltas_Native": Min_Median_Deltas_Native,

            "Max_Range_Deltas_Native": Max_Range_Deltas_Native,
            "Min_Range_Deltas_Native": Min_Range_Deltas_Native,

            "Max_IQR_Deltas_Native": Max_IQR_Deltas_Native,
            "Min_IQR_Deltas_Native": Min_IQR_Deltas_Native,

            "Max_SD_Deltas_Native": Max_SD_Deltas_Native,
            "Min_SD_Deltas_Native": Min_SD_Deltas_Native,

            "Max_Var_Deltas_Native": Max_Var_Deltas_Native,
            "Min_Var_Deltas_Native": Min_Var_Deltas_Native,

            "Max_Skew_Deltas_Native": Max_Skew_Deltas_Native,
            "Min_Skew_Deltas_Native": Min_Skew_Deltas_Native,

            "Max_Kurt_Deltas_Native": Max_Kurt_Deltas_Native,
            "Min_Kurt_Deltas_Native": Min_Kurt_Deltas_Native,

            "Max_Entropy_Deltas_Native": Max_Entropy_Deltas_Native,
            "Min_Entropy_Deltas_Native": Min_Entropy_Deltas_Native,

            "Max_Uniformity_Deltas_Native": Max_Uniformity_Deltas_Native,
            "Min_Uniformity_Deltas_Native": Min_Uniformity_Deltas_Native,

        }

        rows.append(row)


    # RETURN FINAL DATAFRAME

    return pd.DataFrame(rows)








