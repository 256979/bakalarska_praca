

import os
import pydicom
import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

index = [14, 15, 16, 17, 18,
         19, 21, 25, 26, 28,
         30, 31, 32, 33, 34,
         35, 40, 42, 43, 44,
         45, 48, 49, 51, 54,
         60, 62, 70, 72,
         73, 74, 76]

dicom = ["S328460", "S330910", "S331970", "S344360", "S347800",
         "S356830", "S383990", "S468620", "S470440", "S471470",
         "S473760", "S474550", "S482900", "S484100", "S485040",
         "S485710", "S487840", "S490490", "S491740", "S498660",
         "S498680", "S502700", "S504880", "S507220", "S512390",
         "S529070", "S535550", "S569850", "S572420",
         "S573640", "S574210", "S485750"]

TAI_phaase1_r = []
TAI_phaase2_r = []
TAI_phaase3_r = []

TAG_phaase1_r = []
TAG_phaase1_2_r = []
TAG_phaase2_3_r = []
# len(index)
for i in range(len(dicom)):

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

    datasets.sort(key=lambda x: int(x.InstanceNumber))

    metadata = datasets[0]

    slope = getattr(metadata, "RescaleSlope", 1.0)
    intercept = getattr(metadata, "RescaleIntercept", 0.0)
    pixel_spacing = metadata.get("PixelSpacing")
    slice_thickness = metadata.get("SliceThickness")

    slices = []

    for ds in datasets:
        img = ds.pixel_array.astype(np.float32)
        slices.append(img)

    native = np.stack(slices, axis=0)

    # HU
    ct_hu = native * slope + intercept

    import nibabel as nib

    seg_path = rf"/mnt/md0/feketeova/{index[i]}/thrombus_segmentation.nii.gz"

    nii = nib.load(seg_path)
    seg = nii.get_fdata() > 0
    seg = np.transpose(seg, (2, 1, 0))

    print("Segmentation shape:", seg.shape)

    assert ct_hu.shape == seg.shape, "CT and segmentation shape mismatch!"

    mean_density_nativ = ct_hu[seg].mean()

    # phase_1
    phase_1 = rf"/mnt/md0/feketeova/{index[i]}/{index[i]}_phase1_registered.nii.gz"

    nii = nib.load(phase_1)
    data_phase_1 = nii.get_fdata()
    data_phase_1 = np.transpose(data_phase_1, (2, 1, 0))

    assert data_phase_1.shape == seg.shape, "CT and segmentation shape "
    mean_density1 = data_phase_1[seg].mean()

    # phase_2
    phase_2 = rf"/mnt/md0/feketeova/{index[i]}/{index[i]}_phase2_registered.nii.gz"

    nii = nib.load(phase_2)
    data_phase_2 = nii.get_fdata()
    data_phase_2 = np.transpose(data_phase_2, (2, 1, 0))

    assert data_phase_2.shape == seg.shape, "CT and segmentation shape mismatch!"

    mean_density2 = data_phase_2[seg].mean()

    # phase_3
    phase_3 = rf"/mnt/md0/feketeova/{index[i]}/{index[i]}_phase3_registered.nii.gz"

    nii = nib.load(phase_3)
    data_phase_3 = nii.get_fdata()
    data_phase_3 = np.transpose(data_phase_3, (2, 1, 0))

    assert data_phase_3.shape == seg.shape, "CT and segmentation shape mismatch!"

    mean_density3 = data_phase_3[seg].mean()

    max_mean_density = max(mean_density1, mean_density2, mean_density3, mean_density_nativ)
    min_mean_density = min(mean_density1, mean_density2, mean_density3, mean_density_nativ)

    # median
    median_density_nativ = np.median(ct_hu[seg])
    median_density_1 = np.median(data_phase_1[seg])
    median_density_2 = np.median(data_phase_2[seg])
    median_density_3 = np.median(data_phase_3[seg])

    # TAI calculation
    TAI_phaase1 = mean_density1 - mean_density_nativ
    TAI_phaase2 = mean_density2 - mean_density_nativ
    TAI_phaase3 = mean_density3 - mean_density_nativ

    max_TAI = max(TAI_phaase1, TAI_phaase2, TAI_phaase3)
    min_TAI = min(TAI_phaase1, TAI_phaase2, TAI_phaase3)

    # TAG calculation
    TAG_phaase1 = mean_density1 - mean_density_nativ
    TAG_phaase1_2 = mean_density2 - mean_density1
    TAG_phaase2_3 = mean_density3 - mean_density2

    HU_change_1_3 = mean_density3 - mean_density1

    max_TAG = max(TAG_phaase1, TAG_phaase1_2, TAG_phaase2_3)
    min_TAG = min(TAG_phaase1, TAG_phaase1_2, TAG_phaase2_3)

    print(f"Native CT mean HU: {mean_density_nativ:.2f}")
    print(f"Phase 1 mean HU:   {mean_density1:.2f}")
    print(f"Phase 2 mean HU:   {mean_density2:.2f}")
    print(f"Phase 3 mean HU:   {mean_density3:.2f}")

    print(f"TAI Phase 1: {TAI_phaase1:.2f}")
    print(f"TAI Phase 2: {TAI_phaase2:.2f}")
    print(f"TAI Phase 3: {TAI_phaase3:.2f}")

    print(f"TAG Native→P1: {TAG_phaase1:.2f}")
    print(f"TAG P1→P2:     {TAG_phaase1_2:.2f}")
    print(f"TAG P2→P3:     {TAG_phaase2_3:.2f}")
    print("-" * 40)

    # standard deviation
    SD_nativ = ct_hu[seg].std(ddof=1)
    SD_phase_1 = data_phase_1[seg].std(ddof=1)
    SD_phase_2 = data_phase_2[seg].std(ddof=1)
    SD_phase_3 = data_phase_3[seg].std(ddof=1)

    max_SD = max(SD_nativ, SD_phase_1, SD_phase_2, SD_phase_3)
    min_SD = min(SD_nativ, SD_phase_1, SD_phase_2, SD_phase_3)

    print("Native SD:", SD_nativ)
    print("Phase 1 SD:", SD_phase_1)
    print("Phase 2 SD:", SD_phase_2)
    print("Phase 3 SD:", SD_phase_3)

    # change in SD

    SD_change_nativ_to_1 = SD_phase_1 - SD_nativ
    SD_change_nativ_to_2 = SD_phase_2 - SD_nativ
    SD_change_nativ_to_3 = SD_phase_3 - SD_nativ

    max_native_to_phase_SD_change = max(SD_change_nativ_to_1, SD_change_nativ_to_2, SD_change_nativ_to_3)
    min_native_to_phase_SD_change = min(SD_change_nativ_to_1, SD_change_nativ_to_2, SD_change_nativ_to_3)

    SD_change_nativ_to_1 = SD_phase_1 - SD_nativ
    SD_change_1_to_2 = SD_phase_2 - SD_phase_1
    SD_change_2_to_3 = SD_phase_3 - SD_phase_2
    SD_change_1_to_3 = SD_phase_3 - SD_phase_1

    max_adj_phase_SD_change = max(SD_change_1_to_2, SD_change_nativ_to_1, SD_change_2_to_3)
    min_adj_phase_SD_change = min(SD_change_1_to_2, SD_change_nativ_to_1, SD_change_2_to_3)

    print("\nChanges from Native:")
    print("Native → Phase 1:", SD_change_nativ_to_1)
    print("Native → Phase 2:", SD_change_nativ_to_2)
    print("Native → Phase 3:", SD_change_nativ_to_3)

    print("\nChanges between Phases:")
    print("Phase 1 → Phase 2:", SD_change_1_to_2)
    print("Phase 2 → Phase 3:", SD_change_2_to_3)
    print("Phase 1 → Phase 3:", SD_change_1_to_3)

    ''' # Phases stacking
    phases_stack = np.stack([data_phase_1, data_phase_2, data_phase_3], axis=0)
    print(phases_stack.shape)
    # Maximum Intensity Projection
    tMIP = np.max(phases_stack, axis=0)


    # Display results
    plt.figure()
    plt.imshow(tMIP, cmap='gray')
    plt.title("Maximum Intensity Projection (MIP)")
    plt.axis('off')
    plt.show()'''

    # voxel value calculation
    delta_P1_th = data_phase_1[seg] - ct_hu[seg]
    delta_P2_th = data_phase_2[seg] - ct_hu[seg]
    delta_P3_th = data_phase_3[seg] - ct_hu[seg]


    delta_P1_P2_th = data_phase_2[seg] - data_phase_1[seg]
    delta_P2_P3_th = data_phase_3[seg] - data_phase_2[seg]
    delta_P1_P3_th = data_phase_3[seg] - data_phase_1[seg]


    # Statistics voxel

    VoxelMin_Delta_P1 = np.min(delta_P1_th)
    VoxelMax_Delta_P1 = np.max(delta_P1_th)



    VoxelMin_Delta_P2 = np.min(delta_P2_th)
    VoxelMax_Delta_P2 = np.max(delta_P2_th)



    VoxelMin_Delta_P3 = np.min(delta_P3_th)
    VoxelMax_Delta_P3 = np.max(delta_P3_th)


    VoxelMin_Delta_P1_P2 = np.min(delta_P1_P2_th)
    VoxelMax_Delta_P1_P2 = np.max(delta_P1_P2_th)

    VoxelMin_Delta_P2_P3 = np.min(delta_P2_P3_th)
    VoxelMax_Delta_P2_P3 = np.max(delta_P2_P3_th)

    VoxelMin_Delta_P1_P3 = np.min(delta_P1_P3_th)
    VoxelMax_Delta_P1_P3 = np.max(delta_P1_P3_th)


    # Variance


    Var_nativ = np.var(ct_hu[seg], ddof=1)
    Var_phase_1 = np.var(data_phase_1[seg], ddof=1)
    Var_phase_2 = np.var(data_phase_2[seg], ddof=1)
    Var_phase_3 = np.var(data_phase_3[seg], ddof=1)

    print("Native Variance:", Var_nativ)
    print("Phase 1 Variance:", Var_phase_1)
    print("Phase 2 Variance:", Var_phase_2)
    print("Phase 3 Variance:", Var_phase_3)


    # Skewness


    Skew_nativ = skew(ct_hu[seg])
    Skew_phase_1 = skew(data_phase_1[seg])
    Skew_phase_2 = skew(data_phase_2[seg])
    Skew_phase_3 = skew(data_phase_3[seg])

    print("Native Skewness:", Skew_nativ)
    print("Phase 1 Skewness:", Skew_phase_1)
    print("Phase 2 Skewness:", Skew_phase_2)
    print("Phase 3 Skewness:", Skew_phase_3)


    # Kurtosis


    Kurt_nativ = kurtosis(ct_hu[seg])
    Kurt_phase_1 = kurtosis(data_phase_1[seg])
    Kurt_phase_2 = kurtosis(data_phase_2[seg])
    Kurt_phase_3 = kurtosis(data_phase_3[seg])

    print("Native Kurtosis:", Kurt_nativ)
    print("Phase 1 Kurtosis:", Kurt_phase_1)
    print("Phase 2 Kurtosis:", Kurt_phase_2)
    print("Phase 3 Kurtosis:", Kurt_phase_3)


    # Entropy


    hist_native, _ = np.histogram(ct_hu[seg], bins=256, density=True)
    hist_native = hist_native + 1e-12
    Entropy_nativ = -np.sum(hist_native * np.log2(hist_native))

    hist_p1, _ = np.histogram(data_phase_1[seg], bins=256, density=True)
    hist_p1 = hist_p1 + 1e-12
    Entropy_phase_1 = -np.sum(hist_p1 * np.log2(hist_p1))

    hist_p2, _ = np.histogram(data_phase_2[seg], bins=256, density=True)
    hist_p2 = hist_p2 + 1e-12
    Entropy_phase_2 = -np.sum(hist_p2 * np.log2(hist_p2))

    hist_p3, _ = np.histogram(data_phase_3[seg], bins=256, density=True)
    hist_p3 = hist_p3 + 1e-12
    Entropy_phase_3 = -np.sum(hist_p3 * np.log2(hist_p3))

    print("Native Entropy:", Entropy_nativ)
    print("Phase 1 Entropy:", Entropy_phase_1)
    print("Phase 2 Entropy:", Entropy_phase_2)
    print("Phase 3 Entropy:", Entropy_phase_3)


    # Uniformity


    Uniformity_nativ = np.sum(hist_native ** 2)
    Uniformity_phase_1 = np.sum(hist_p1 ** 2)
    Uniformity_phase_2 = np.sum(hist_p2 ** 2)
    Uniformity_phase_3 = np.sum(hist_p3 ** 2)

    print("Native Uniformity:", Uniformity_nativ)
    print("Phase 1 Uniformity:", Uniformity_phase_1)
    print("Phase 2 Uniformity:", Uniformity_phase_2)
    print("Phase 3 Uniformity:", Uniformity_phase_3)


    # Variance of HU changes


    Var_Delta_P1 = np.var(delta_P1_th, ddof=1)
    Var_Delta_P2 = np.var(delta_P2_th, ddof=1)
    Var_Delta_P3 = np.var(delta_P3_th, ddof=1)

    Var_Delta_P1_P2 = np.var(delta_P1_P2_th, ddof=1)
    Var_Delta_P2_P3 = np.var(delta_P2_P3_th, ddof=1)
    Var_Delta_P1_P3 = np.var(delta_P1_P3_th, ddof=1)

    # Skewness of HU changes


    Skew_Delta_P1 = skew(delta_P1_th)
    Skew_Delta_P2 = skew(delta_P2_th)
    Skew_Delta_P3 = skew(delta_P3_th)

    Skew_Delta_P1_P2 = skew(delta_P1_P2_th)
    Skew_Delta_P2_P3 = skew(delta_P2_P3_th)
    Skew_Delta_P1_P3 = skew(delta_P1_P3_th)


    # Kurtosis of HU changes


    Kurt_Delta_P1 = kurtosis(delta_P1_th)
    Kurt_Delta_P2 = kurtosis(delta_P2_th)
    Kurt_Delta_P3 = kurtosis(delta_P3_th)

    Kurt_Delta_P1_P2 = kurtosis(delta_P1_P2_th)
    Kurt_Delta_P2_P3 = kurtosis(delta_P2_P3_th)
    Kurt_Delta_P1_P3 = kurtosis(delta_P1_P3_th)


    # Entropy of HU changes


    hist_d1, _ = np.histogram(delta_P1_th, bins=256, density=True)
    hist_d1 = hist_d1 + 1e-12
    Entropy_Delta_P1 = -np.sum(hist_d1 * np.log2(hist_d1))

    hist_d2, _ = np.histogram(delta_P2_th, bins=256, density=True)
    hist_d2 = hist_d2 + 1e-12
    Entropy_Delta_P2 = -np.sum(hist_d2 * np.log2(hist_d2))

    hist_d3, _ = np.histogram(delta_P3_th, bins=256, density=True)
    hist_d3 = hist_d3 + 1e-12
    Entropy_Delta_P3 = -np.sum(hist_d3 * np.log2(hist_d3))


    # Uniformity of HU changes


    Uniformity_Delta_P1 = np.sum(hist_d1 ** 2)
    Uniformity_Delta_P2 = np.sum(hist_d2 ** 2)
    Uniformity_Delta_P3 = np.sum(hist_d3 ** 2)

    # rate of change
    # time to peak
    import pandas as pd

    csv_path = "/home/feketeova/Documents/results.csv"


    # feature dictionary


    patient_data = {
        "PatientIndex": index[i],
        "DICOM_ID": dicom[i],

        "Mean_Native": mean_density_nativ,
        "Mean_Phase1": mean_density1,
        "Mean_Phase2": mean_density2,
        "Mean_Phase3": mean_density3,
        "Max_mean_density": max_mean_density,
        "Min_mean_density": min_mean_density,

        "median_density_nativ": median_density_nativ,
        "Median_density_1": median_density_1,
        "Median_density_2": median_density_2,
        "Median_density_3": median_density_3,

        "TAI_Phase1": TAI_phaase1,
        "TAI_Phase2": TAI_phaase2,
        "TAI_Phase3": TAI_phaase3,
        "Max_TAI": max_TAI,
        "Min_TAI": min_TAI,

        "TAG_Native_P1": TAG_phaase1,
        "TAG_P1_P2": TAG_phaase1_2,
        "TAG_P2_P3": TAG_phaase2_3,
        "Max_TAG": max_TAG,
        "Min_TAG": min_TAG,
        "HU_change_1_3": HU_change_1_3,

        "SD_Native": SD_nativ,
        "SD_Phase1": SD_phase_1,
        "SD_Phase2": SD_phase_2,
        "SD_Phase3": SD_phase_3,
        "max_SD": max_SD,
        "min_SD": min_SD,

        "SD_Native_P1": SD_change_nativ_to_1,
        "SD_Native_P2": SD_change_nativ_to_2,
        "SD_Native_P3": SD_change_nativ_to_3,
        "Max_native_to_phase_SD_change": max_native_to_phase_SD_change,
        "Min_native_to_phase_SD_change": min_native_to_phase_SD_change,

        "SD_P1_P2": SD_change_1_to_2,
        "SD_P2_P3": SD_change_2_to_3,
        "SD_P1_P3": SD_change_1_to_3,
        "Max_adj_phase_SD_change": max_adj_phase_SD_change,
        "Min_adj_phase_SD_change": min_adj_phase_SD_change,

        "VoxelMin_Delta_P1": VoxelMin_Delta_P1,
        "VoxelMax_Delta_P1": VoxelMax_Delta_P1,

        "VoxelMin_Delta_P2": VoxelMin_Delta_P2,
        "VoxelMax_Delta_P2": VoxelMax_Delta_P2,

        "VoxelMin_Delta_P3": VoxelMin_Delta_P3,
        "VoxelMax_Delta_P3": VoxelMax_Delta_P3,

        "VoxelMin_Delta_P1_P2": VoxelMin_Delta_P1_P2,
        "VoxelMax_Delta_P1_P2": VoxelMax_Delta_P1_P2,

        "VoxelMin_Delta_P2_P3": VoxelMin_Delta_P2_P3,
        "VoxelMax_Delta_P2_P3": VoxelMax_Delta_P2_P3,

        "VoxelMin_Delta_P1_P3": VoxelMin_Delta_P1_P3,
        "VoxelMax_Delta_P1_P3": VoxelMax_Delta_P1_P3,

        "Variance_Native": Var_nativ,
        "Variance_Phase1": Var_phase_1,
        "Variance_Phase2": Var_phase_2,
        "Variance_Phase3": Var_phase_3,


        "Skewness_Native": Skew_nativ,
        "Skewness_Phase1": Skew_phase_1,
        "Skewness_Phase2": Skew_phase_2,
        "Skewness_Phase3": Skew_phase_3,


        "Kurtosis_Native": Kurt_nativ,
        "Kurtosis_Phase1": Kurt_phase_1,
        "Kurtosis_Phase2": Kurt_phase_2,
        "Kurtosis_Phase3": Kurt_phase_3,


        "Entropy_Native": Entropy_nativ,
        "Entropy_Phase1": Entropy_phase_1,
        "Entropy_Phase2": Entropy_phase_2,
        "Entropy_Phase3": Entropy_phase_3,


        "Uniformity_Native": Uniformity_nativ,
        "Uniformity_Phase1": Uniformity_phase_1,
        "Uniformity_Phase2": Uniformity_phase_2,
        "Uniformity_Phase3": Uniformity_phase_3,


        "Variance_Delta_P1": Var_Delta_P1,
        "Variance_Delta_P2": Var_Delta_P2,
        "Variance_Delta_P3": Var_Delta_P3,


        "Skewness_Delta_P1": Skew_Delta_P1,
        "Skewness_Delta_P2": Skew_Delta_P2,
        "Skewness_Delta_P3": Skew_Delta_P3,


        "Kurtosis_Delta_P1": Kurt_Delta_P1,
        "Kurtosis_Delta_P2": Kurt_Delta_P2,
        "Kurtosis_Delta_P3": Kurt_Delta_P3,


        "Entropy_Delta_P1": Entropy_Delta_P1,
        "Entropy_Delta_P2": Entropy_Delta_P2,
        "Entropy_Delta_P3": Entropy_Delta_P3,


        "Uniformity_Delta_P1": Uniformity_Delta_P1,
        "Uniformity_Delta_P2": Uniformity_Delta_P2,
        "Uniformity_Delta_P3": Uniformity_Delta_P3,
    }

    new_row = pd.DataFrame([patient_data])



    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:

        df_existing = pd.read_csv(csv_path)

        # new features
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
            new_row = new_row[df_existing.columns]
            df_final = pd.concat([df_existing, new_row], ignore_index=True)

    else:
        print("Creating new CSV file.")
        df_final = new_row

    df_final.to_csv(csv_path, index=False)
    print("Saved successfully.")

