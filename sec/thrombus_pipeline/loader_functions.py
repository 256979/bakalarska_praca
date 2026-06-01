import os
import glob
import re
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
import SimpleITK as sitk




# DICOM LOADER
'''
dicom_dir: str – path to the folder containing the DICOM series
data: NumPy array (x, y, z) – voxel intensities in canonical RAS+ orientation
aff2axcodes(img.affine): tuple – anatomical orientation codes of the array
This function:
    Loads a full DICOM series and reconstructs it into a 3D volume
    Converts the DICOM voxel array from (z,y,x) to (x,y,z)
    Extracts the dicom geometry
    Builds the affine matrix
    Converts from LPS to RAS
    Creates a NIfTI image
    Reorientation into canonical RAS+
    Returns the voxel data and axis codes
'''
def load_dicom_series(dicom_dir):

    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(dicom_dir)

    if len(files) == 0:
        raise RuntimeError(f"No DICOM files found in: {dicom_dir}")

    reader.SetFileNames(files)
    img_sitk = reader.Execute()

    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    arr = np.transpose(arr, (2, 1, 0))

    spacing = np.array(img_sitk.GetSpacing())
    direction = np.array(img_sitk.GetDirection()).reshape(3, 3)
    origin = np.array(img_sitk.GetOrigin())

    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction * spacing
    affine_lps[:3, 3] = origin

    LPS_to_RAS = np.diag([-1, -1, 1, 1])
    affine_ras = LPS_to_RAS @ affine_lps

    img = nib.Nifti1Image(arr, affine_ras)
    img = nib.as_closest_canonical(img)

    data = np.asarray(img.dataobj, dtype=np.float32)

    return data, aff2axcodes(img.affine)



# NIFTI LOADER

'''
nifti_path: str – path to a .nii or .nii.gz NIfTI file
data: NumPy array – voxel intensities in RAS+ orientation
aff2axcodes(img.affine): tuple – anatomical orientation codes of the array

This function:
    Loads a NIfTI file using nibabel
    Checks whether the image is already in RAS+ orientation
    Reorients the image into canonical RAS+ if necessary
    Returns the voxel data and orientation axis codes

This function loads a NIfTI file and ensures that it is in cannoniacl RAS+ orientation
'''
def load_nifti(nifti_path):

    img = nib.load(nifti_path)
    codes = aff2axcodes(img.affine)

    if codes != ('R', 'A', 'S'):
        img = nib.as_closest_canonical(img)

    data = np.asarray(img.dataobj, dtype=np.float32)

    return data, aff2axcodes(img.affine)



# PHASE NUMBER EXTRACTION

'''
path: str – file path to a NIfTI file
output: int or None – extracted phase number, or None if not found

This function:
    Extracts the phase number from a NIfTI filename
    Ignores files containing segmentation-related keywords
    Handles filenames with or without leading numeric prefixes
    Returns the detected phase number or None if no number is found
'''
def extract_phase_number(path):

    name = os.path.basename(path).lower()

    if any(w in name for w in ["seg", "mask", "label"]):
        return None

    parts = name.split("_")

    if len(parts) > 1 and parts[0].isdigit():
        name = "_".join(parts[1:])

    phase_number = re.search(r"(\d+)", name)

    if phase_number:
        return int(phase_number.group(1))

    return None



# PATIENT DISCOVERY

'''
root: str – str – dataset root directory (provided via config.toml)
output: list[str] – sorted list of patient folder paths

This function:
    Searches the root directory for patient folders
    Identifies valid patients by folder names consisting only of digits
    Returns a sorted list of all matching patient directories
'''
def find_patients(root):
    return sorted(
        [
            d for d in glob.glob(os.path.join(root, "*"))
            if os.path.isdir(d) and os.path.basename(d).isdigit()
        ],
        key=lambda p: int(os.path.basename(p))
    )


# FIND DICOM

'''
patient_dir: str – path to the patient folder
output: str(dicom_path) or None – path to the native DICOM directory, or None if not found

This function:
    Recursively searches the patient directory
    Identifies the native DICOM folder by matching the name "nativ", "native"
    Returns the path to the DICOM directory or None if missing
'''

def find_dicom(patient_dir):

    for dicom_path, dirs, files in os.walk(patient_dir):
        name = os.path.basename(dicom_path).lower()
        if name == "nativ" or name == "native":
            return dicom_path

    return None


# LOAD CT

'''
dicom_dir: str - path to the native DICOM directory
output: (ct_hu, ct_codes) or (None, None)

This function:
     Loads the CT volume by calling load_dicom_series()
     Returns the voxel array (HU values) and orientation codes
     Returns (None, None) if loading fails
'''
def load_ct(dicom_dir):

    try:
        ct_hu, ct_codes = load_dicom_series(dicom_dir)
        return ct_hu, ct_codes
    except Exception:
        return None, None



# SEGMENTATION

'''
patient_dir: str – path to the folder containing patient data
output:
    seg: NumPy array – binary segmentation mask
    seg_codes: tuple – orientation axis codes
    (None, None) if segmentation is missing, empty, or fails to load

This function:
    Searches the patient folder for a segmentation file
    Loads the segmentation volume by calling load_nifti()
    Converts the segmentation to a binary mask
    Validates that the mask contains at least one positive voxel
    Returns (None, None) if the segmentation is missing or empty
'''

def find_and_load_segmentation(patient_dir):

    seg_possible = glob.glob(os.path.join(patient_dir, "*seg*.nii*"))

    if len(seg_possible) == 0:
        return None, None

    try:
        seg, seg_codes = load_nifti(seg_possible[0])
        seg = seg > 0

        if not np.any(seg):
            return None, None

        return seg, seg_codes

    except Exception:
        return None, None



# PHASES

'''
patient_dir: str – path to the patient folder
seg: NumPy array – segmentation mask used for validation
seg_codes: tuple – orientation codes of the segmentation
output: dict or None – {"P1": array, "P2": array, "P3": array} or None if loading fails

This function:
    Searches the patient folder for NIfTI files representing contrast phases
    Extracts the phase number from each filename using extract_phase_number()
    Sorts phases by their numeric order 
    Loads each phase volume by calling load_nifti()
    Checks shape and orientation consistency with the segmentation
    Collects valid phases into a dictionary keyed as "P1", "P2", "P3"
    Returns the dictionary of loaded phases or None if loading fails
'''
def find_and_load_phases(patient_dir, seg, seg_codes):

    all_nifti = glob.glob(os.path.join(patient_dir, "*.nii*"))

    phases = []

    for f in all_nifti:
        phase_num = extract_phase_number(f)
        if phase_num is None:
            continue
        phases.append((phase_num, f))

    phases.sort()

    phase_data = {}

    for phase_num, phase_path in phases:

        try:
            data, codes = load_nifti(phase_path)
        except Exception:
            continue

        if data.shape != seg.shape:
            continue

        if codes != seg_codes:
            continue

        phase_data[f"P{phase_num}"] = data

    if not all(k in phase_data for k in ["P1", "P2", "P3"]):
        return None

    return phase_data




# FULL DATA LOADER
'''
patient_dir: str – path to the patient folder
output: dict containing:
        "patient_id": int
        "vox_Native": 1D NumPy array – thrombus voxels (native CT)
        "vox_P1":     1D NumPy array – thrombus voxels (phase 1)
        "vox_P2":     1D NumPy array – thrombus voxels (phase 2)
        "vox_P3":     1D NumPy array – thrombus voxels (phase 3)
        "full_Native": 3D NumPy array – full native CT volume
        "full_P1":     3D NumPy array – full phase 1 volume
        "full_P2":     3D NumPy array – full phase 2 volume
        "full_P3":     3D NumPy array – full phase 3 volume
        "seg_mask":    3D NumPy array – binary thrombus segmentation mask

This function:
    Loads the native CT DICOM series by calling find_dicom()
    Loads the CT volume by calling load_ct()
    Loads the thrombus segmentation by calling find_and_load_segmentation()
    Verifies CT/segmentation shape and orientation consistency
    Loads all required contrast phases by calling find_and_load_phases()
    Extracts voxel intensities inside the segmentation mask for CT and all phases
    Returns a dictionary containing patient ID, masked voxel values, full volumes,
    and the segmentation mask
    Returns None if any loading, alignment, or phase requirement fails
'''

def load_patient_data(patient_dir):

    patient_id = os.path.basename(patient_dir)
    print("Patient ID " + patient_id)

    # CT
    dicom_dir = find_dicom(patient_dir)
    if dicom_dir is None:
        print(f"  ERROR: No DICOM directory found for patient {patient_id}")
        return None

    ct_result = load_ct(dicom_dir)
    if ct_result is None or ct_result[0] is None:
        print(f"  ERROR: Failed to load CT for patient {patient_id}")
        return None

    ct_hu, ct_codes = ct_result

    # SEG
    seg_result = find_and_load_segmentation(patient_dir)
    if seg_result is None or seg_result[0] is None:
        print(f"  ERROR: Missing or empty segmentation for patient {patient_id}")
        return None

    seg, seg_codes = seg_result

    if ct_hu.shape != seg.shape:
        print(f"  ERROR: Shape mismatch CT vs segmentation for patient {patient_id}")
        print(f"         CT shape:  {ct_hu.shape}")
        print(f"         SEG shape: {seg.shape}")
        return None

    if ct_codes != seg_codes:
        print(f"  ERROR: Orientation mismatch CT vs segmentation for patient {patient_id}")
        print(f"         CT codes:  {ct_codes}")
        print(f"         SEG codes: {seg_codes}")
        return None

    # PHASES
    phase_data = find_and_load_phases(patient_dir, seg, seg_codes)
    if phase_data is None:
        print(f"  ERROR: Missing or invalid phases (P1/P2/P3) for patient {patient_id}")
        return None

    # MASKS
    vox_Native = ct_hu[seg]
    vox_P1 = phase_data["P1"][seg]
    vox_P2 = phase_data["P2"][seg]
    vox_P3 = phase_data["P3"][seg]

    return {
        "patient_id": int(patient_id),
        "vox_Native": vox_Native,
        "vox_P1": vox_P1,
        "vox_P2": vox_P2,
        "vox_P3": vox_P3,
        "full_Native": ct_hu,
        "full_P1": phase_data["P1"],
        "full_P2": phase_data["P2"],
        "full_P3": phase_data["P3"],
        "seg_mask": seg
    }
