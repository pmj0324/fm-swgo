"""
Plot distributions of each label dimension (mccoreX, mccoreY, mczenithAngle,
mcazimuthAngle, mclogEnergy) from the dataset.
Uses h5py bulk read for speed. Adjust the variables below and run.
"""
from pathlib import Path

import hdf5plugin  # noqa: F401  # HDF5 compression
import h5py
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Input: edit these and run the script
# -----------------------------------------------------------------------------
DATA_PATHS = [
    "/store/hawc/swgo/ml/M7/gammaD9v40_trainmoreUHE2.hdf5",
]
MAX_EVENTS = None  # None = use all events; set to e.g. 50000 for a quick plot
OUT_DIR = "."
OUT_NAME = "label_distribution.png"
# Cut: eventnHit >= CUT_NHIT; mccoreR < CUT_MCCORER only (None = no cut)
CUT_NHIT = 15   # e.g. 5 to require at least 5 hits
CUT_MCCORER = 150  # keep only events with mccoreR < this value
# -----------------------------------------------------------------------------

LABEL_NAMES = [
    "mccoreX",
    "mccoreY",
    "mczenithAngle",
    "mcazimuthAngle",
    "mclogEnergy",
]
LABEL_UNITS = ["m", "m", "rad", "rad", ""]


def load_labels_bulk(
    data_paths: list,
    max_events: int | None,
    cut_nhit: float | None,
    cut_mccorer: float | None,
) -> tuple[np.ndarray, int]:
    """Bulk-read label fields from HDF5 data/data. Apply cut: eventnHit >= cut_nhit,
    mccoreR < cut_mccorer only (None = no cut).
    Returns (labels array (N, 5), n_events_after_cut).
    """
    chunks = []
    n_read = 0
    for path in data_paths:
        with h5py.File(path, "r") as f:
            d = f["data/data"]
            n_file = d.shape[0]
            if max_events is not None:
                need = max_events - n_read
                if need <= 0:
                    break
                n_slice = min(n_file, need)
            else:
                n_slice = n_file
            data = d[:n_slice]
        n_read += n_slice
        # Cut: eventnHit >= cut_nhit; mccoreR < cut_mccorer only
        mask = np.ones(n_slice, dtype=bool)
        if cut_nhit is not None:
            mask &= np.asarray(data["eventnHit"]) >= cut_nhit
        if cut_mccorer is not None:
            mask &= np.asarray(data["mccoreR"]) < cut_mccorer
        # Structured array -> (n_slice, 5), then apply mask
        chunk = np.column_stack([
            data["mccoreX"],
            data["mccoreY"],
            data["mczenithAngle"],
            data["mcazimuthAngle"],
            data["mclogEnergy"],
        ])
        chunks.append(chunk[mask])
        if max_events is not None and n_read >= max_events:
            break
    labels = np.concatenate(chunks, axis=0)
    return labels, labels.shape[0]


labels, n_use = load_labels_bulk(
    DATA_PATHS, MAX_EVENTS, CUT_NHIT, CUT_MCCORER
)
cut_str = f" (cut: eventnHit>={CUT_NHIT}, mccoreR<{CUT_MCCORER})" if (CUT_NHIT is not None or CUT_MCCORER is not None) else ""
print(f"Loaded {n_use} labels{cut_str}.\n")

# Print actual min/max (and percentiles) per dimension so we can verify x-axis
print("Per-label min / max / 1% / 99% (data range):")
for k in range(5):
    col = labels[:, k]
    mn, mx = col.min(), col.max()
    p1 = np.percentile(col, 1)
    p99 = np.percentile(col, 99)
    print(f"  {LABEL_NAMES[k]:18s}: min={mn:12.4f}  max={mx:12.4f}  1%={p1:8.4f}  99%={p99:8.4f}")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for k in range(5):
    ax = axes[k]
    col = labels[:, k]
    mn, mx = col.min(), col.max()
    # x-axis = data range so plot reflects actual min/max
    if mx > mn:
        ax.set_xlim(mn, mx)
    else:
        ax.set_xlim(mn - 0.5, mx + 0.5)  # single value: small margin so one bin visible
    ax.hist(col, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    unit = LABEL_UNITS[k]
    ax.set_xlabel(f"{LABEL_NAMES[k]}" + (f" [{unit}]" if unit else ""))
    ax.set_ylabel("count")
    ax.set_title(LABEL_NAMES[k])

axes[5].axis("off")

fig.suptitle(f"Label distributions (n={n_use})", fontsize=12)
fig.tight_layout()

out_path = Path(OUT_DIR) / OUT_NAME
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")
