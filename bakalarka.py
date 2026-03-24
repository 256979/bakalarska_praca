
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import nibabel as nib
from nibabel.orientations import aff2axcodes
import pandas as pd

index = [14,15,16,17,18,
         19,21,25,26,28,
         30,31,32,33,34,
         35,40,42,43,44,
         45,48,49,51,54,
         60,62,70,72,
         73,74,76]

dicom = ["S328460", "S330910" , "S331970", "S344360","S347800",
         "S356830", "S383990", "S468620", "S470440", "S471470",
         "S473760", "S474550", "S482900", "S484100", "S485040",
         "S485710", "S487840", "S490490", "S491740", "S498660",
         "S498680", "S502700", "S504880", "S507220", "S512390",
         "S529070", "S535550", "S569850", "S572420",
         "S573640", "S574210", "S485750"]


for i in range(len(dicom)):

    # -------- LOAD DICOM --------
    dicom_dir = rf"/mnt/md0/feketeova/{index[i]}/Export/DICOM/{dicom[i]}/nativ"

    dicom_files = []
    for f in os.listdir(dicom_dir):
        path = os.path.join(dicom_dir, f)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            dicom_files.append(path)
        except Exception:
            pass

    datasets = []
    for f in dicom_files:
        ds = pydicom.dcmread(f)
        datasets.append(ds)

    datasets.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    metadata  = datasets[0]
    slope     = float(getattr(metadata, "RescaleSlope",     1.0))
    intercept = float(getattr(metadata, "RescaleIntercept", 0.0))

    slices = []
    for ds in datasets:
        img = ds.pixel_array.astype(np.float32)
        slices.append(img)

    native_raw = np.stack(slices, axis=-1)
    ct_hu      = native_raw * slope + intercept

    # -------- BUILD PROPER AFFINE FROM DICOM TAGS --------
    # Row and column cosines from ImageOrientationPatient
    iop         = np.array(metadata.ImageOrientationPatient, dtype=float)
    row_cosine  = iop[:3]
    col_cosine  = iop[3:]
    slice_cosine = np.cross(row_cosine, col_cosine)

    pixel_spacing   = [float(x) for x in metadata.PixelSpacing]
    slice_thickness = float(getattr(metadata, "SliceThickness", 1.0))
    origin          = np.array(metadata.ImagePositionPatient, dtype=float)

    affine          = np.eye(4)
    affine[:3, 0]   = row_cosine   * pixel_spacing[1]   # column direction
    affine[:3, 1]   = col_cosine   * pixel_spacing[0]   # row direction
    affine[:3, 2]   = slice_cosine * slice_thickness     # slice direction
    affine[:3, 3]   = origin                             # position of first voxel

    ct_img = nib.Nifti1Image(ct_hu, affine)
    ct_img = nib.as_closest_canonical(ct_img)
    ct_hu  = ct_img.get_fdata()

    print(f"Patient {index[i]} — CT orientation: {aff2axcodes(ct_img.affine)}  shape: {ct_hu.shape}")

    # -------- SEGMENTATION --------
    seg_path = rf"/mnt/md0/feketeova/{index[i]}/thrombus_segmentation.nii.gz"
    seg_img  = nib.load(seg_path)
    seg_img  = nib.as_closest_canonical(seg_img)
    seg      = seg_img.get_fdata() > 0

    print(f"Patient {index[i]} — Seg orientation: {aff2axcodes(seg_img.affine)}  shape: {seg.shape}")

    # Shape check
    assert ct_hu.shape == seg.shape, \
        f"Patient {index[i]}: CT shape {ct_hu.shape} != Seg shape {seg.shape}"

    # Orientation check — catches axis flips that shape alone cannot detect
    ct_codes  = aff2axcodes(ct_img.affine)
    seg_codes = aff2axcodes(seg_img.affine)
    assert ct_codes == seg_codes, \
        f"Patient {index[i]}: CT orientation {ct_codes} != Seg orientation {seg_codes}"

    # -------- LOAD PHASES --------
    def _load_phase(path, label, seg_shape, seg_codes):
        """Load a NIfTI phase, reorient to canonical, and verify shape + orientation."""
        img  = nib.load(path)
        img  = nib.as_closest_canonical(img)
        data = img.get_fdata()
        codes = aff2axcodes(img.affine)
        assert data.shape == seg_shape, \
            f"{label} shape {data.shape} != Seg shape {seg_shape}"
        assert codes == seg_codes, \
            f"{label} orientation {codes} != Seg orientation {seg_codes}"
        return data

    data_P1 = _load_phase(
        rf"/mnt/md0/feketeova/{index[i]}/{index[i]}_phase1_registered.nii.gz",
        f"Patient {index[i]} Phase1", seg.shape, seg_codes)

    data_P2 = _load_phase(
        rf"/mnt/md0/feketeova/{index[i]}/{index[i]}_phase2_registered.nii.gz",
        f"Patient {index[i]} Phase2", seg.shape, seg_codes)

    data_P3 = _load_phase(
        rf"/mnt/md0/feketeova/{index[i]}/{index[i]}_phase3_registered.nii.gz",
        f"Patient {index[i]} Phase3", seg.shape, seg_codes)

    # Masked voxel arrays — used for all statistics below
    vox_Native = ct_hu[seg]
    vox_P1     = data_P1[seg]
    vox_P2     = data_P2[seg]
    vox_P3     = data_P3[seg]

    # ================================
    # DELTA ARRAYS (voxel-level HU changes)
    # Naming: Delta_<from>_<to>
    # ================================
    vox_Delta_N_P1  = vox_P1 - vox_Native   # Native → Phase 1
    vox_Delta_N_P2  = vox_P2 - vox_Native   # Native → Phase 2
    vox_Delta_N_P3  = vox_P3 - vox_Native   # Native → Phase 3
    vox_Delta_P1_P2 = vox_P2 - vox_P1       # Phase 1 → Phase 2
    vox_Delta_P2_P3 = vox_P3 - vox_P2       # Phase 2 → Phase 3
    vox_Delta_P1_P3 = vox_P3 - vox_P1       # Phase 1 → Phase 3

    # ================================
    # HELPER: entropy and uniformity
    # ================================
    def _entropy(arr):
        h, _ = np.histogram(arr, bins=256, density=True)
        h = h + 1e-12
        return -np.sum(h * np.log2(h))

    def _uniformity(arr):
        h, _ = np.histogram(arr, bins=256, density=True)
        h = h + 1e-12
        return np.sum(h ** 2)

    # ================================
    # MEAN
    # ================================
    Mean_Native = vox_Native.mean()
    Mean_P1     = vox_P1.mean()
    Mean_P2     = vox_P2.mean()
    Mean_P3     = vox_P3.mean()
    Max_Mean    = max(Mean_Native, Mean_P1, Mean_P2, Mean_P3)
    Min_Mean    = min(Mean_Native, Mean_P1, Mean_P2, Mean_P3)

    print(f"Mean  Native:{Mean_Native:.2f}  P1:{Mean_P1:.2f}  P2:{Mean_P2:.2f}  P3:{Mean_P3:.2f}")

    # ================================
    # MEDIAN
    # ================================
    Median_Native = np.median(vox_Native)
    Median_P1     = np.median(vox_P1)
    Median_P2     = np.median(vox_P2)
    Median_P3     = np.median(vox_P3)
    Max_Median    = max(Median_Native, Median_P1, Median_P2, Median_P3)
    Min_Median    = min(Median_Native, Median_P1, Median_P2, Median_P3)

    # ================================
    # MIN INTENSITY
    # ================================
    Min_Native = np.min(vox_Native)
    Min_P1     = np.min(vox_P1)
    Min_P2     = np.min(vox_P2)
    Min_P3     = np.min(vox_P3)
    Max_Min    = max(Min_Native, Min_P1, Min_P2, Min_P3)
    Min_Min    = min(Min_Native, Min_P1, Min_P2, Min_P3)

    # ================================
    # MAX INTENSITY
    # ================================
    Max_Native = np.max(vox_Native)
    Max_P1     = np.max(vox_P1)
    Max_P2     = np.max(vox_P2)
    Max_P3     = np.max(vox_P3)
    Max_Max    = max(Max_Native, Max_P1, Max_P2, Max_P3)
    Min_Max    = min(Max_Native, Max_P1, Max_P2, Max_P3)

    # ================================
    # RANGE
    # ================================
    Range_Native     = Max_Native - Min_Native
    Range_P1         = Max_P1     - Min_P1
    Range_P2         = Max_P2     - Min_P2
    Range_P3         = Max_P3     - Min_P3
    Max_Range        = max(Range_Native, Range_P1, Range_P2, Range_P3)
    Min_Range        = min(Range_Native, Range_P1, Range_P2, Range_P3)

    Range_Delta_N_P1  = np.max(vox_Delta_N_P1)  - np.min(vox_Delta_N_P1)
    Range_Delta_N_P2  = np.max(vox_Delta_N_P2)  - np.min(vox_Delta_N_P2)
    Range_Delta_N_P3  = np.max(vox_Delta_N_P3)  - np.min(vox_Delta_N_P3)
    Range_Delta_P1_P2 = np.max(vox_Delta_P1_P2) - np.min(vox_Delta_P1_P2)
    Range_Delta_P2_P3 = np.max(vox_Delta_P2_P3) - np.min(vox_Delta_P2_P3)
    Range_Delta_P1_P3 = np.max(vox_Delta_P1_P3) - np.min(vox_Delta_P1_P3)

    # ================================
    # IQR
    # ================================
    IQR_Native     = np.percentile(vox_Native, 75) - np.percentile(vox_Native, 25)
    IQR_P1         = np.percentile(vox_P1,     75) - np.percentile(vox_P1,     25)
    IQR_P2         = np.percentile(vox_P2,     75) - np.percentile(vox_P2,     25)
    IQR_P3         = np.percentile(vox_P3,     75) - np.percentile(vox_P3,     25)
    Max_IQR        = max(IQR_Native, IQR_P1, IQR_P2, IQR_P3)
    Min_IQR        = min(IQR_Native, IQR_P1, IQR_P2, IQR_P3)

    IQR_Delta_N_P1  = np.percentile(vox_Delta_N_P1,  75) - np.percentile(vox_Delta_N_P1,  25)
    IQR_Delta_N_P2  = np.percentile(vox_Delta_N_P2,  75) - np.percentile(vox_Delta_N_P2,  25)
    IQR_Delta_N_P3  = np.percentile(vox_Delta_N_P3,  75) - np.percentile(vox_Delta_N_P3,  25)
    IQR_Delta_P1_P2 = np.percentile(vox_Delta_P1_P2, 75) - np.percentile(vox_Delta_P1_P2, 25)
    IQR_Delta_P2_P3 = np.percentile(vox_Delta_P2_P3, 75) - np.percentile(vox_Delta_P2_P3, 25)
    IQR_Delta_P1_P3 = np.percentile(vox_Delta_P1_P3, 75) - np.percentile(vox_Delta_P1_P3, 25)

    # ================================
    # TAI  (Thrombus Attenuation Increase vs Native)
    # ================================
    TAI_P1  = Mean_P1 - Mean_Native
    TAI_P2  = Mean_P2 - Mean_Native
    TAI_P3  = Mean_P3 - Mean_Native
    Max_TAI = max(TAI_P1, TAI_P2, TAI_P3)
    Min_TAI = min(TAI_P1, TAI_P2, TAI_P3)

    print(f"TAI   P1:{TAI_P1:.2f}  P2:{TAI_P2:.2f}  P3:{TAI_P3:.2f}")

    # ================================
    # TAG  (between adjacent phases)
    # ================================
    TAG_N_P1  = Mean_P1 - Mean_Native
    TAG_P1_P2 = Mean_P2 - Mean_P1
    TAG_P2_P3 = Mean_P3 - Mean_P2
    TAG_P1_P3 = Mean_P3 - Mean_P1
    Max_TAG   = max(TAG_N_P1, TAG_P1_P2, TAG_P2_P3)
    Min_TAG   = min(TAG_N_P1, TAG_P1_P2, TAG_P2_P3)

    print(f"TAG   N→P1:{TAG_N_P1:.2f}  P1→P2:{TAG_P1_P2:.2f}  P2→P3:{TAG_P2_P3:.2f}  P1→P3:{TAG_P1_P3:.2f}")
    print("-" * 40)

    # ================================
    # STANDARD DEVIATION
    # ================================
    SD_Native = vox_Native.std(ddof=1)
    SD_P1     = vox_P1.std(ddof=1)
    SD_P2     = vox_P2.std(ddof=1)
    SD_P3     = vox_P3.std(ddof=1)
    Max_SD    = max(SD_Native, SD_P1, SD_P2, SD_P3)
    Min_SD    = min(SD_Native, SD_P1, SD_P2, SD_P3)

    # SD changes: Native → each phase
    SD_Delta_N_P1  = SD_P1 - SD_Native
    SD_Delta_N_P2  = SD_P2 - SD_Native
    SD_Delta_N_P3  = SD_P3 - SD_Native
    Max_SD_Delta_N = max(SD_Delta_N_P1, SD_Delta_N_P2, SD_Delta_N_P3)
    Min_SD_Delta_N = min(SD_Delta_N_P1, SD_Delta_N_P2, SD_Delta_N_P3)

    # SD changes: between adjacent phases
    SD_Delta_P1_P2   = SD_P2 - SD_P1
    SD_Delta_P2_P3   = SD_P3 - SD_P2
    SD_Delta_P1_P3   = SD_P3 - SD_P1
    Max_SD_Delta_Adj = max(SD_Delta_N_P1, SD_Delta_P1_P2, SD_Delta_P2_P3)
    Min_SD_Delta_Adj = min(SD_Delta_N_P1, SD_Delta_P1_P2, SD_Delta_P2_P3)

    print(f"SD    Native:{SD_Native:.2f}  P1:{SD_P1:.2f}  P2:{SD_P2:.2f}  P3:{SD_P3:.2f}")

    # ================================
    # VOXEL MIN / MAX OF DELTA ARRAYS
    # ================================
    VoxMin_Delta_N_P1  = np.min(vox_Delta_N_P1)
    VoxMax_Delta_N_P1  = np.max(vox_Delta_N_P1)
    VoxMin_Delta_N_P2  = np.min(vox_Delta_N_P2)
    VoxMax_Delta_N_P2  = np.max(vox_Delta_N_P2)
    VoxMin_Delta_N_P3  = np.min(vox_Delta_N_P3)
    VoxMax_Delta_N_P3  = np.max(vox_Delta_N_P3)
    VoxMin_Delta_P1_P2 = np.min(vox_Delta_P1_P2)
    VoxMax_Delta_P1_P2 = np.max(vox_Delta_P1_P2)
    VoxMin_Delta_P2_P3 = np.min(vox_Delta_P2_P3)
    VoxMax_Delta_P2_P3 = np.max(vox_Delta_P2_P3)
    VoxMin_Delta_P1_P3 = np.min(vox_Delta_P1_P3)
    VoxMax_Delta_P1_P3 = np.max(vox_Delta_P1_P3)

    # ================================
    # VARIANCE
    # ================================
    Var_Native     = np.var(vox_Native, ddof=1)
    Var_P1         = np.var(vox_P1,     ddof=1)
    Var_P2         = np.var(vox_P2,     ddof=1)
    Var_P3         = np.var(vox_P3,     ddof=1)
    Max_Var        = max(Var_Native, Var_P1, Var_P2, Var_P3)
    Min_Var        = min(Var_Native, Var_P1, Var_P2, Var_P3)

    Var_Delta_N_P1  = np.var(vox_Delta_N_P1,  ddof=1)
    Var_Delta_N_P2  = np.var(vox_Delta_N_P2,  ddof=1)
    Var_Delta_N_P3  = np.var(vox_Delta_N_P3,  ddof=1)
    Var_Delta_P1_P2 = np.var(vox_Delta_P1_P2, ddof=1)
    Var_Delta_P2_P3 = np.var(vox_Delta_P2_P3, ddof=1)
    Var_Delta_P1_P3 = np.var(vox_Delta_P1_P3, ddof=1)

    print(f"Var   Native:{Var_Native:.2f}  P1:{Var_P1:.2f}  P2:{Var_P2:.2f}  P3:{Var_P3:.2f}")

    # ================================
    # SKEWNESS
    # ================================
    Skew_Native     = skew(vox_Native)
    Skew_P1         = skew(vox_P1)
    Skew_P2         = skew(vox_P2)
    Skew_P3         = skew(vox_P3)
    Max_Skew        = max(Skew_Native, Skew_P1, Skew_P2, Skew_P3)
    Min_Skew        = min(Skew_Native, Skew_P1, Skew_P2, Skew_P3)

    Skew_Delta_N_P1  = skew(vox_Delta_N_P1)
    Skew_Delta_N_P2  = skew(vox_Delta_N_P2)
    Skew_Delta_N_P3  = skew(vox_Delta_N_P3)
    Skew_Delta_P1_P2 = skew(vox_Delta_P1_P2)
    Skew_Delta_P2_P3 = skew(vox_Delta_P2_P3)
    Skew_Delta_P1_P3 = skew(vox_Delta_P1_P3)

    print(f"Skew  Native:{Skew_Native:.4f}  P1:{Skew_P1:.4f}  P2:{Skew_P2:.4f}  P3:{Skew_P3:.4f}")

    # ================================
    # KURTOSIS
    # ================================
    Kurt_Native     = kurtosis(vox_Native)
    Kurt_P1         = kurtosis(vox_P1)
    Kurt_P2         = kurtosis(vox_P2)
    Kurt_P3         = kurtosis(vox_P3)
    Max_Kurt        = max(Kurt_Native, Kurt_P1, Kurt_P2, Kurt_P3)
    Min_Kurt        = min(Kurt_Native, Kurt_P1, Kurt_P2, Kurt_P3)

    Kurt_Delta_N_P1  = kurtosis(vox_Delta_N_P1)
    Kurt_Delta_N_P2  = kurtosis(vox_Delta_N_P2)
    Kurt_Delta_N_P3  = kurtosis(vox_Delta_N_P3)
    Kurt_Delta_P1_P2 = kurtosis(vox_Delta_P1_P2)
    Kurt_Delta_P2_P3 = kurtosis(vox_Delta_P2_P3)
    Kurt_Delta_P1_P3 = kurtosis(vox_Delta_P1_P3)

    print(f"Kurt  Native:{Kurt_Native:.4f}  P1:{Kurt_P1:.4f}  P2:{Kurt_P2:.4f}  P3:{Kurt_P3:.4f}")

    # ================================
    # ENTROPY
    # ================================
    Entropy_Native     = _entropy(vox_Native)
    Entropy_P1         = _entropy(vox_P1)
    Entropy_P2         = _entropy(vox_P2)
    Entropy_P3         = _entropy(vox_P3)
    Max_Entropy        = max(Entropy_Native, Entropy_P1, Entropy_P2, Entropy_P3)
    Min_Entropy        = min(Entropy_Native, Entropy_P1, Entropy_P2, Entropy_P3)

    Entropy_Delta_N_P1  = _entropy(vox_Delta_N_P1)
    Entropy_Delta_N_P2  = _entropy(vox_Delta_N_P2)
    Entropy_Delta_N_P3  = _entropy(vox_Delta_N_P3)
    Entropy_Delta_P1_P2 = _entropy(vox_Delta_P1_P2)
    Entropy_Delta_P2_P3 = _entropy(vox_Delta_P2_P3)
    Entropy_Delta_P1_P3 = _entropy(vox_Delta_P1_P3)

    print(f"Entr  Native:{Entropy_Native:.4f}  P1:{Entropy_P1:.4f}  P2:{Entropy_P2:.4f}  P3:{Entropy_P3:.4f}")

    # ================================
    # UNIFORMITY
    # ================================
    Uniformity_Native     = _uniformity(vox_Native)
    Uniformity_P1         = _uniformity(vox_P1)
    Uniformity_P2         = _uniformity(vox_P2)
    Uniformity_P3         = _uniformity(vox_P3)
    Max_Uniformity        = max(Uniformity_Native, Uniformity_P1, Uniformity_P2, Uniformity_P3)
    Min_Uniformity        = min(Uniformity_Native, Uniformity_P1, Uniformity_P2, Uniformity_P3)

    Uniformity_Delta_N_P1  = _uniformity(vox_Delta_N_P1)
    Uniformity_Delta_N_P2  = _uniformity(vox_Delta_N_P2)
    Uniformity_Delta_N_P3  = _uniformity(vox_Delta_N_P3)
    Uniformity_Delta_P1_P2 = _uniformity(vox_Delta_P1_P2)
    Uniformity_Delta_P2_P3 = _uniformity(vox_Delta_P2_P3)
    Uniformity_Delta_P1_P3 = _uniformity(vox_Delta_P1_P3)

    print(f"Unif  Native:{Uniformity_Native:.6f}  P1:{Uniformity_P1:.6f}  P2:{Uniformity_P2:.6f}  P3:{Uniformity_P3:.6f}")

    # ==============================
    # BUILD FEATURE DICTIONARY
    # Convention:
    #   <Statistic>_<Phase>           for per-phase values
    #   <Statistic>_Delta_<from>_<to> for voxel-level delta values
    #   Max/Min_<Statistic>           for cross-phase aggregates
    # ==============================
    patient_data = {
        "PatientIndex": index[i],
        "DICOM_ID":     dicom[i],

        # --- Mean ---
        "Mean_Native":   Mean_Native,
        "Mean_P1":       Mean_P1,
        "Mean_P2":       Mean_P2,
        "Mean_P3":       Mean_P3,
        "Max_Mean":      Max_Mean,
        "Min_Mean":      Min_Mean,

        # --- Median ---
        "Median_Native": Median_Native,
        "Median_P1":     Median_P1,
        "Median_P2":     Median_P2,
        "Median_P3":     Median_P3,
        "Max_Median":    Max_Median,
        "Min_Median":    Min_Median,

        # --- Min intensity ---
        "Min_Native":    Min_Native,
        "Min_P1":        Min_P1,
        "Min_P2":        Min_P2,
        "Min_P3":        Min_P3,
        "Max_Min":       Max_Min,
        "Min_Min":       Min_Min,

        # --- Max intensity ---
        "Max_Native":    Max_Native,
        "Max_P1":        Max_P1,
        "Max_P2":        Max_P2,
        "Max_P3":        Max_P3,
        "Max_Max":       Max_Max,
        "Min_Max":       Min_Max,

        # --- Range ---
        "Range_Native":      Range_Native,
        "Range_P1":          Range_P1,
        "Range_P2":          Range_P2,
        "Range_P3":          Range_P3,
        "Max_Range":         Max_Range,
        "Min_Range":         Min_Range,
        "Range_Delta_N_P1":  Range_Delta_N_P1,
        "Range_Delta_N_P2":  Range_Delta_N_P2,
        "Range_Delta_N_P3":  Range_Delta_N_P3,
        "Range_Delta_P1_P2": Range_Delta_P1_P2,
        "Range_Delta_P2_P3": Range_Delta_P2_P3,
        "Range_Delta_P1_P3": Range_Delta_P1_P3,

        # --- IQR ---
        "IQR_Native":      IQR_Native,
        "IQR_P1":          IQR_P1,
        "IQR_P2":          IQR_P2,
        "IQR_P3":          IQR_P3,
        "Max_IQR":         Max_IQR,
        "Min_IQR":         Min_IQR,
        "IQR_Delta_N_P1":  IQR_Delta_N_P1,
        "IQR_Delta_N_P2":  IQR_Delta_N_P2,
        "IQR_Delta_N_P3":  IQR_Delta_N_P3,
        "IQR_Delta_P1_P2": IQR_Delta_P1_P2,
        "IQR_Delta_P2_P3": IQR_Delta_P2_P3,
        "IQR_Delta_P1_P3": IQR_Delta_P1_P3,

        # --- TAI ---
        "TAI_P1":        TAI_P1,
        "TAI_P2":        TAI_P2,
        "TAI_P3":        TAI_P3,
        "Max_TAI":       Max_TAI,
        "Min_TAI":       Min_TAI,

        # --- TAG ---
        "TAG_N_P1":      TAG_N_P1,
        "TAG_P1_P2":     TAG_P1_P2,
        "TAG_P2_P3":     TAG_P2_P3,
        "TAG_P1_P3":     TAG_P1_P3,
        "Max_TAG":       Max_TAG,
        "Min_TAG":       Min_TAG,

        # --- Standard Deviation ---
        "SD_Native":         SD_Native,
        "SD_P1":             SD_P1,
        "SD_P2":             SD_P2,
        "SD_P3":             SD_P3,
        "Max_SD":            Max_SD,
        "Min_SD":            Min_SD,
        "SD_Delta_N_P1":     SD_Delta_N_P1,
        "SD_Delta_N_P2":     SD_Delta_N_P2,
        "SD_Delta_N_P3":     SD_Delta_N_P3,
        "Max_SD_Delta_N":    Max_SD_Delta_N,
        "Min_SD_Delta_N":    Min_SD_Delta_N,
        "SD_Delta_P1_P2":    SD_Delta_P1_P2,
        "SD_Delta_P2_P3":    SD_Delta_P2_P3,
        "SD_Delta_P1_P3":    SD_Delta_P1_P3,
        "Max_SD_Delta_Adj":  Max_SD_Delta_Adj,
        "Min_SD_Delta_Adj":  Min_SD_Delta_Adj,


        # --- Voxel Min / Max of delta arrays ---
        "VoxMin_Delta_N_P1":  VoxMin_Delta_N_P1,
        "VoxMax_Delta_N_P1":  VoxMax_Delta_N_P1,
        "VoxMin_Delta_N_P2":  VoxMin_Delta_N_P2,
        "VoxMax_Delta_N_P2":  VoxMax_Delta_N_P2,
        "VoxMin_Delta_N_P3":  VoxMin_Delta_N_P3,
        "VoxMax_Delta_N_P3":  VoxMax_Delta_N_P3,
        "VoxMin_Delta_P1_P2": VoxMin_Delta_P1_P2,
        "VoxMax_Delta_P1_P2": VoxMax_Delta_P1_P2,
        "VoxMin_Delta_P2_P3": VoxMin_Delta_P2_P3,
        "VoxMax_Delta_P2_P3": VoxMax_Delta_P2_P3,
        "VoxMin_Delta_P1_P3": VoxMin_Delta_P1_P3,
        "VoxMax_Delta_P1_P3": VoxMax_Delta_P1_P3,

        # --- Variance ---
        "Var_Native":      Var_Native,
        "Var_P1":          Var_P1,
        "Var_P2":          Var_P2,
        "Var_P3":          Var_P3,
        "Max_Var":         Max_Var,
        "Min_Var":         Min_Var,
        "Var_Delta_N_P1":  Var_Delta_N_P1,
        "Var_Delta_N_P2":  Var_Delta_N_P2,
        "Var_Delta_N_P3":  Var_Delta_N_P3,
        "Var_Delta_P1_P2": Var_Delta_P1_P2,
        "Var_Delta_P2_P3": Var_Delta_P2_P3,
        "Var_Delta_P1_P3": Var_Delta_P1_P3,

        # --- Skewness ---
        "Skew_Native":         Skew_Native,
        "Skew_P1":             Skew_P1,
        "Skew_P2":             Skew_P2,
        "Skew_P3":             Skew_P3,
        "Max_Skew":            Max_Skew,
        "Min_Skew":            Min_Skew,
        "Skew_Delta_N_P1":     Skew_Delta_N_P1,
        "Skew_Delta_N_P2":     Skew_Delta_N_P2,
        "Skew_Delta_N_P3":     Skew_Delta_N_P3,
        "Skew_Delta_P1_P2":    Skew_Delta_P1_P2,
        "Skew_Delta_P2_P3":    Skew_Delta_P2_P3,
        "Skew_Delta_P1_P3":    Skew_Delta_P1_P3,

        # --- Kurtosis ---
        "Kurt_Native":         Kurt_Native,
        "Kurt_P1":             Kurt_P1,
        "Kurt_P2":             Kurt_P2,
        "Kurt_P3":             Kurt_P3,
        "Max_Kurt":            Max_Kurt,
        "Min_Kurt":            Min_Kurt,
        "Kurt_Delta_N_P1":     Kurt_Delta_N_P1,
        "Kurt_Delta_N_P2":     Kurt_Delta_N_P2,
        "Kurt_Delta_N_P3":     Kurt_Delta_N_P3,
        "Kurt_Delta_P1_P2":    Kurt_Delta_P1_P2,
        "Kurt_Delta_P2_P3":    Kurt_Delta_P2_P3,
        "Kurt_Delta_P1_P3":    Kurt_Delta_P1_P3,

        # --- Entropy ---
        "Entropy_Native":      Entropy_Native,
        "Entropy_P1":          Entropy_P1,
        "Entropy_P2":          Entropy_P2,
        "Entropy_P3":          Entropy_P3,
        "Max_Entropy":         Max_Entropy,
        "Min_Entropy":         Min_Entropy,
        "Entropy_Delta_N_P1":  Entropy_Delta_N_P1,
        "Entropy_Delta_N_P2":  Entropy_Delta_N_P2,
        "Entropy_Delta_N_P3":  Entropy_Delta_N_P3,
        "Entropy_Delta_P1_P2": Entropy_Delta_P1_P2,
        "Entropy_Delta_P2_P3": Entropy_Delta_P2_P3,
        "Entropy_Delta_P1_P3": Entropy_Delta_P1_P3,

        # --- Uniformity ---
        "Uniformity_Native":      Uniformity_Native,
        "Uniformity_P1":          Uniformity_P1,
        "Uniformity_P2":          Uniformity_P2,
        "Uniformity_P3":          Uniformity_P3,
        "Max_Uniformity":         Max_Uniformity,
        "Min_Uniformity":         Min_Uniformity,
        "Uniformity_Delta_N_P1":  Uniformity_Delta_N_P1,
        "Uniformity_Delta_N_P2":  Uniformity_Delta_N_P2,
        "Uniformity_Delta_N_P3":  Uniformity_Delta_N_P3,
        "Uniformity_Delta_P1_P2": Uniformity_Delta_P1_P2,
        "Uniformity_Delta_P2_P3": Uniformity_Delta_P2_P3,
        "Uniformity_Delta_P1_P3": Uniformity_Delta_P1_P3,
    }

    new_row = pd.DataFrame([patient_data])

    # ==========================================
    # SAVE TO CSV (add/update, never duplicate)
    # ==========================================
    csv_path = "/home/feketeova/Documents/results_a.csv"

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:

        df_existing = pd.read_csv(csv_path)

        # Add any new columns that appear for the first time
        for col in new_row.columns:
            if col not in df_existing.columns:
                df_existing[col] = np.nan

        if index[i] in df_existing["PatientIndex"].values:
            print(f"Patient {index[i]} exists → updating missing values.")
            patient_mask = df_existing["PatientIndex"] == index[i]
            for col in new_row.columns:
                current_value = df_existing.loc[patient_mask, col].values[0]
                if pd.isna(current_value):
                    df_existing.loc[patient_mask, col] = new_row[col].values[0]
            df_final = df_existing

        else:
            print(f"Adding new patient {index[i]}")
            new_row  = new_row[df_existing.columns]
            df_final = pd.concat([df_existing, new_row], ignore_index=True)

    else:
        print("Creating new CSV file.")
        df_final = new_row

    df_final.to_csv(csv_path, index=False)
    print("Saved successfully.")
