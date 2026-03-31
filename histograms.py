import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.orientations import aff2axcodes
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

PLOT_DIR = "/home/feketeova/Documents/plots/histograms"
os.makedirs(PLOT_DIR, exist_ok=True)


def _load_phase(path, label, seg_shape, seg_codes):
    img = nib.as_closest_canonical(nib.load(path))
    data = img.get_fdata()
    codes = aff2axcodes(img.affine)
    assert data.shape == seg_shape, f"{label} shape {data.shape} != Seg shape {seg_shape}"
    assert codes == seg_codes, f"{label} orientation {codes} != Seg orientation {seg_codes}"
    return data


for i in range(len(dicom)):
    pid = index[i]
    print(f"Patient {pid} ...", end=" ", flush=True)

    dicom_dir = rf"/mnt/md0/feketeova/{pid}/Export/DICOM/{dicom[i]}/nativ"
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_dir))
    img_sitk = reader.Execute()
    native_lps = np.transpose(sitk.GetArrayFromImage(img_sitk).astype(np.float32), (2, 1, 0))

    spacing = np.array(img_sitk.GetSpacing())
    direction = np.array(img_sitk.GetDirection()).reshape(3, 3)
    origin = np.array(img_sitk.GetOrigin())
    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction * spacing
    affine_lps[:3, 3] = origin
    affine_ras = np.diag([-1, -1, 1, 1]) @ affine_lps

    ct_img = nib.as_closest_canonical(nib.Nifti1Image(native_lps, affine_ras))
    ct_hu = ct_img.get_fdata()
    print(np.min(ct_hu), np.max(ct_hu))

    # Segmentation
    seg_img = nib.as_closest_canonical(
        nib.load(rf"/mnt/md0/feketeova/{pid}/thrombus_segmentation.nii.gz"))
    seg = seg_img.get_fdata() > 0
    seg_codes = aff2axcodes(seg_img.affine)

    assert ct_hu.shape == seg.shape
    assert aff2axcodes(ct_img.affine) == seg_codes

    # Phases
    base = rf"/mnt/md0/feketeova/{pid}"
    data_P1 = _load_phase(f"{base}/{pid}_phase1_registered.nii.gz", f"{pid} P1", seg.shape, seg_codes)
    data_P2 = _load_phase(f"{base}/{pid}_phase2_registered.nii.gz", f"{pid} P2", seg.shape, seg_codes)
    data_P3 = _load_phase(f"{base}/{pid}_phase3_registered.nii.gz", f"{pid} P3", seg.shape, seg_codes)

    # thrombus voxels
    vox_Native = ct_hu[seg]
    vox_P1 = data_P1[seg]
    vox_P2 = data_P2[seg]
    vox_P3 = data_P3[seg]

    # histogram
    phases = [vox_Native, vox_P1, vox_P2, vox_P3]
    labels = ['Native', 'Phase 1', 'Phase 2', 'Phase 3']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    fig.suptitle(f"Patient {pid} — HU distribution within thrombus mask",
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, arr, label, color in zip(axes, phases, labels, colors):
        ax.hist(arr, bins=80, color=color, edgecolor='white', linewidth=0.3, alpha=0.9)
        ax.axvline(arr.mean(), color='black', linestyle='--',
                   linewidth=1.4, label=f"Mean: {arr.mean():.1f}")
        ax.axvline(np.median(arr), color='dimgray', linestyle=':',
                   linewidth=1.4, label=f"Median: {np.median(arr):.1f}")
        ax.set_title(label, fontsize=12, fontweight='bold', color=color)
        ax.set_xlabel("HU value", fontsize=10)
        ax.set_ylabel("Voxel count", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.7)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"patient_{pid}_HU_histogram.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved → {out_path}")