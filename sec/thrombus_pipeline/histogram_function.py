import os
import numpy as np
import matplotlib.pyplot as plt

#HISTOGRAM GENERATION
'''
vox_native: 1D NumPy array – HU values inside the thrombus mask (native CT)
vox_p1:     1D NumPy array – HU values inside the thrombus mask (phase 1)
vox_p2:     1D NumPy array – HU values inside the thrombus mask (phase 2)
vox_p3:     1D NumPy array – HU values inside the thrombus mask (phase 3)
patient_id: int – patient identifier
out_dir:    str – directory where the histogram figure will be saved (from config.toml)

This function:
    Creates a 2×2 histogram figure showing HU distributions for:
        Native, Phase 1, Phase 2, Phase 3
    Computes and overlays the mean and median HU values for each phase
    Formats each subplot with titles, labels, grid, and styling
    Saves the resulting figure as a PNG file in the specified output directory
    Returns the full path to the saved histogram image

'''

def plot_histograms(vox_native, vox_p1, vox_p2, vox_p3, patient_id, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    phases = [vox_native, vox_p1, vox_p2, vox_p3]
    labels = ['Native', 'Phase 1', 'Phase 2', 'Phase 3']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    global_min = min(arr.min() for arr in phases)
    global_max = max(arr.max() for arr in phases)

    fig, axes = plt.subplots(2, 2, figsize=(50, 25), sharey=False)
    axes = axes.flatten()

    for ax, arr, label, color in zip(axes, phases, labels, colors):
        ax.hist(
            arr,
            bins=80,
            range=(global_min, global_max),
            color=color,
            edgecolor='white',
            linewidth=0.3,
            alpha=0.9
        )

        ax.axvline(arr.mean(), color='black', linestyle='--',
                   linewidth=1.7, label=f"Mean: {arr.mean():.1f}")
        ax.axvline(np.median(arr), color='dimgray', linestyle=':',
                   linewidth=1.4, label=f"Median: {np.median(arr):.1f}")

        ax.set_title(label, fontsize=40, fontweight='bold', color=color)
        ax.set_xlabel("HU value", fontsize=40)
        ax.set_ylabel("Voxel count", fontsize=40)
        ax.legend(fontsize=40, framealpha=0.7)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=40)

    plt.tight_layout()

    out_path = os.path.join(out_dir, f"patient_{patient_id}_HU_histogram.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return out_path
