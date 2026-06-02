import os
import numpy as np
import matplotlib.pyplot as plt

from loader_functions import load_patient_data, find_patients



# RENDER SLICE

'''
vol: NumPy array – full 3D CT volume for a given phase
seg: NumPy array – binary thrombus segmentation mask
z: int – axial slice index to render
canvas_size: int – size of the square output canvas (pixels)

This function:
    Extracts the segmentation mask for slice z.
    Masks the CT slice so only thrombus voxels remain (background = NaN).
    Computes the bounding box of the thrombus.
    Crops to the bounding box.
    Centers the cropped region on a square canvas.
    Returns the centered 2D slice as a NumPy array.
'''
def _render_slice(vol, seg, z, canvas_size):

    mask = seg[:, :, z]
    slice_masked = np.where(mask, vol[:, :, z], np.nan)

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    crop = slice_masked[y0:y1+1, x0:x1+1]
    h, w = crop.shape

    target = int(canvas_size * 0.75)
    scale = target / max(h, w)

    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    crop_resized = np.full((new_h, new_w), np.nan)
    yy = (np.linspace(0, h - 1, new_h)).astype(int)
    xx = (np.linspace(0, w - 1, new_w)).astype(int)
    crop_resized[:, :] = crop[np.ix_(yy, xx)]

    canvas = np.full((canvas_size, canvas_size), np.nan)
    y_off = (canvas_size - new_h) // 2
    x_off = (canvas_size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = crop_resized

    return canvas



# EXPORT THROMBUS SLICES

'''
ROOT: str – dataset root directory (from config.toml)
OUT_DIR: str – output directory for slice exports (from config.toml)

This function:
    - Finds all patient folders.
    - Loads CT volumes + segmentation.
    - Identifies slices containing thrombus.
    - Computes a consistent canvas size.
    - For each phase (Native, P1, P2, P3):
        * Creates subfolder phase_name/ with normal PNGs
        * Renders each slice using _render_slice()
        * Saves normal slices only (no annotation)
    - Skips patients with missing data.
    - Prints progress and completion messages.
'''
def export_thrombus_slices(ROOT, OUT_DIR):

    os.makedirs(OUT_DIR, exist_ok=True)
    patient_dirs = find_patients(ROOT)

    for patient_dir in patient_dirs:

        data = load_patient_data(patient_dir)
        if data is None:
            continue

        pid = data["patient_id"]
        seg = data["seg_mask"]

        full_native = data["full_Native"]
        full_p1 = data["full_P1"]
        full_p2 = data["full_P2"]
        full_p3 = data["full_P3"]

        print(f"\n=== Patient {pid} ===")

        valid_slices = np.where(seg.sum(axis=(0, 1)) > 0)[0]
        if len(valid_slices) == 0:
            print("No valid slices.")
            continue

        max_dim = 0
        for z in valid_slices:
            coords = np.where(seg[:, :, z])
            h = coords[0].max() - coords[0].min() + 1
            w = coords[1].max() - coords[1].min() + 1
            max_dim = max(max_dim, h, w)

        canvas_size = int(np.ceil(max_dim / 0.75))

        patient_out = os.path.join(OUT_DIR, f"patient_{pid}")
        os.makedirs(patient_out, exist_ok=True)

        phases = {
            "Native": full_native,
            "Phase1": full_p1,
            "Phase2": full_p2,
            "Phase3": full_p3
        }

        for phase_name, vol in phases.items():

            phase_dir = os.path.join(patient_out, phase_name)
            os.makedirs(phase_dir, exist_ok=True)

            print(f"  Processing {phase_name}")

            for z in valid_slices:

                rendered = _render_slice(vol, seg, z, canvas_size)
                if rendered is None:
                    continue

                out_png = os.path.join(phase_dir, f"slice_{z:03d}.png")

                fig = plt.figure(figsize=(8, 8), facecolor="#FFF9D6")
                ax = plt.gca()
                ax.set_facecolor("#FFF9D6")

                cmap = plt.cm.gray.copy()
                cmap.set_bad("#FFF9D6")

                plt.imshow(rendered, cmap=cmap, interpolation="nearest")
                plt.title(f"Patient {pid} | {phase_name} | Slice {z}", fontsize=14)
                plt.axis("off")

                plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.08)
                plt.close(fig)

    print("\nDONE exporting slices.")



# HEAD SLICE VISUALISATION

'''
native, P1, P2, P3 : 3D NumPy arrays
    Full CT volumes for native and contrast phases.
SLICE : int
    Axial slice index to export.
patient_id : int or str
    Patient identifier for labeling.
out_dir : str
    Output directory where the figure will be saved.

This function:
    Extracts the same axial slice from:
        - Native CT
        - Phase 1
        - Phase 2
        - Phase 3
    Arranges them side‑by‑side in a 4‑panel figure.
    Saves the figure to disk (PNG format).
    Does NOT display the figure on screen.
    Intended for quick visual inspection of alignment and quality.
'''
def show_raw_slices(native, P1, P2, P3, SLICE, patient_id, out_dir):

    slice_native = native[:, :, SLICE]
    slice_p1 = P1[:, :, SLICE]
    slice_p2 = P2[:, :, SLICE]
    slice_p3 = P3[:, :, SLICE]

    fig, ax = plt.subplots(1, 4, figsize=(25, 6))

    ax[0].imshow(slice_native, cmap="gray")
    ax[0].set_title(f"Native (Patient {patient_id})")
    ax[0].axis("off")

    ax[1].imshow(slice_p1, cmap="gray")
    ax[1].set_title(f"Phase 1 (Patient {patient_id})")
    ax[1].axis("off")

    ax[2].imshow(slice_p2, cmap="gray")
    ax[2].set_title(f"Phase 2 (Patient {patient_id})")
    ax[2].axis("off")

    ax[3].imshow(slice_p3, cmap="gray")
    ax[3].set_title(f"Phase 3 (Patient {patient_id})")
    ax[3].axis("off")

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"patient_{patient_id}_slice_{SLICE}.png")
    plt.savefig(save_path, dpi=150)

    plt.close(fig)
