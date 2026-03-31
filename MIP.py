import os

import numpy as np

from scipy.stats import skew, kurtosis
import nibabel as nib
from nibabel.orientations import aff2axcodes
import pandas as pd
import SimpleITK as sitk

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

for i in range(len(dicom)):


    dicom_dir = rf"/mnt/md0/feketeova/{index[i]}/Export/DICOM/{dicom[i]}/nativ"

    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(files)
    img_sitk = reader.Execute()


    native_lps = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # (z, y, x)


    native_lps = np.transpose(native_lps, (2, 1, 0))  # (x, y, z)


    spacing = np.array(img_sitk.GetSpacing())  # (x, y, z)
    direction = np.array(img_sitk.GetDirection()).reshape(3, 3)  # 3x3
    origin = np.array(img_sitk.GetOrigin())  # (x, y, z)

    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction * spacing
    affine_lps[:3, 3] = origin

    # Convert LPS → RAS
    LPS_to_RAS = np.diag([-1, -1, 1, 1])
    affine_ras = LPS_to_RAS @ affine_lps

    ct_img = nib.Nifti1Image(native_lps, affine_ras)
    ct_img = nib.as_closest_canonical(ct_img)
    ct_hu = ct_img.get_fdata()

    print(f"Patient {index[i]} — CT orientation: {aff2axcodes(ct_img.affine)}  shape: {ct_hu.shape}")

    # -------- SEGMENTATION --------
    seg_path = rf"/mnt/md0/feketeova/{index[i]}/thrombus_segmentation.nii.gz"
    seg_img = nib.load(seg_path)
    seg_img = nib.as_closest_canonical(seg_img)
    seg = seg_img.get_fdata() > 0

    print(f"Patient {index[i]} — Seg orientation: {aff2axcodes(seg_img.affine)}  shape: {seg.shape}")

    # Shape check
    assert ct_hu.shape == seg.shape, \
        f"Patient {index[i]}: CT shape {ct_hu.shape} != Seg shape {seg.shape}"

    # Orientation check
    ct_codes = aff2axcodes(ct_img.affine)
    seg_codes = aff2axcodes(seg_img.affine)
    assert ct_codes == seg_codes, \
        f"Patient {index[i]}: CT orientation {ct_codes} != Seg orientation {seg_codes}"


    # -------- LOAD PHASES --------
    def _load_phase(path, label, seg_shape, seg_codes):
        """Load a NIfTI phase, reorient to canonical, and verify shape + orientation."""
        img = nib.load(path)
        img = nib.as_closest_canonical(img)
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



    # --- 3D MIP across phases (P1, P2, P3) ---
    Proj_MIP_3D = np.maximum.reduce([data_P1, data_P2, data_P3])

    # --- 3D MinIP across phases (P1, P2, P3) ---
    Proj_MinIP_3D = np.minimum.reduce([data_P1, data_P2, data_P3])


    vox_MIP = Proj_MIP_3D[seg]
    vox_MinIP = Proj_MinIP_3D[seg]

    # MIP STATISTICS

    Proj_MIP_Mean = float(np.mean(vox_MIP))
    Proj_MIP_Std = float(np.std(vox_MIP))
    Proj_MIP_Skew = float(skew(vox_MIP))
    Proj_MIP_Kurt = float(kurtosis(vox_MIP))
    Proj_MIP_P5 = float(np.percentile(vox_MIP, 5))
    Proj_MIP_P95 = float(np.percentile(vox_MIP, 95))
    Proj_MIP_IQR = float(np.percentile(vox_MIP, 75) - np.percentile(vox_MIP, 25))

    # MinIP STATISTICS

    Proj_MinIP_Mean = float(np.mean(vox_MinIP))
    Proj_MinIP_Std = float(np.std(vox_MinIP))
    Proj_MinIP_Skew = float(skew(vox_MinIP))
    Proj_MinIP_Kurt = float(kurtosis(vox_MinIP))
    Proj_MinIP_P5 = float(np.percentile(vox_MinIP, 5))
    Proj_MinIP_P95 = float(np.percentile(vox_MinIP, 95))
    Proj_MinIP_IQR = float(np.percentile(vox_MinIP, 75) - np.percentile(vox_MinIP, 25))

    # PROJECTION RANGE (3D)

    Proj_3D_Range = float(np.max(vox_MIP) - np.min(vox_MinIP))


    # BUILD FEATURE DICTIONARY

    patient_data = {
        "PatientIndex": index[i],
        "DICOM_ID": dicom[i],

        "Proj_MIP_3D_Mean": Proj_MIP_Mean,
        "Proj_MIP_3D_Std": Proj_MIP_Std,
        "Proj_MIP_3D_Skew": Proj_MIP_Skew,
        "Proj_MIP_3D_Kurt": Proj_MIP_Kurt,
        "Proj_MIP_3D_P5": Proj_MIP_P5,
        "Proj_MIP_3D_P95": Proj_MIP_P95,
        "Proj_MIP_3D_IQR": Proj_MIP_IQR,

        "Proj_MinIP_3D_Mean": Proj_MinIP_Mean,
        "Proj_MinIP_3D_Std": Proj_MinIP_Std,
        "Proj_MinIP_3D_Skew": Proj_MinIP_Skew,
        "Proj_MinIP_3D_Kurt": Proj_MinIP_Kurt,
        "Proj_MinIP_3D_P5": Proj_MinIP_P5,
        "Proj_MinIP_3D_P95": Proj_MinIP_P95,
        "Proj_MinIP_3D_IQR": Proj_MinIP_IQR,

        "Proj_3D_Range": Proj_3D_Range,

    }

    new_row = pd.DataFrame([patient_data])


    # SAVE TO CSV

    csv_path = "/home/feketeova/Documents/MIP.csv"

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:

        df_existing = pd.read_csv(csv_path)


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
