import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Use the same dataloader as train.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader.h5_loader import H5EventDataset


def load_event(h5_path: str, event_idx: int, position_path: str | None = None):
    """
    Load a single event using H5EventDataset with angle_convergence=False
    so labels are [mccoreX, mccoreY, mczenithAngle, mcazimuthAngle, mclogEnergy].
    Returns x, y, z (PMT positions), time, charge, label (length 5).
    """
    dataset = H5EventDataset(
        file_paths=[h5_path],
        position_path=position_path,
        angle_convergence=False,
    )
    sample = dataset[event_idx]
    x_tensor = sample["x"]  # (722, 5)
    label_tensor = sample["y"]  # (5,)
    x_np = x_tensor.numpy()
    label = label_tensor.numpy()
    time = x_np[:, 0]
    charge = x_np[:, 1]
    x = x_np[:, 2]
    y = x_np[:, 3]
    z = x_np[:, 4]
    return x, y, z, time, charge, label


def split_top_bottom_by_z(z: np.ndarray):
    """
    Split PMTs into two groups (top / bottom) based on z coordinate.

    We assume there are exactly two distinct z levels corresponding to
    the upper and lower PMTs on each tank.
    """
    unique_z = np.unique(z)
    if unique_z.size != 2:
        raise ValueError(
            f"Expected exactly 2 unique z-levels for top/bottom PMTs, "
            f"but found {unique_z.size}: {unique_z}"
        )

    z_low, z_high = np.sort(unique_z)  # In this geometry: 0.06 = top, 0.07 = bottom
    top_mask = z == z_low
    bottom_mask = z == z_high
    return (bottom_mask, z_high), (top_mask, z_low)


# Colormaps: different but similar feel (both sequential, perceptually uniform)
CMAP_TIME = "inferno"   # black → red → yellow (time)
CMAP_CHARGE = "plasma"  # purple → yellow (NPE)

def _add_colorbar_same_height(ax, scatter, label: str) -> None:
    """Add a colorbar to ax with the same height as the plot rectangle."""
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(scatter, cax=cax, label=label)


def _scatter_tanks(ax, x: np.ndarray, y: np.ndarray, values: np.ndarray, norm=None, cmap: str = "viridis"):
    """
    Draw thin black circle outline at every tank position; fill with color only where value != 0.
    """
    # 1) Thin black circle at every (x, y)
    ax.scatter(
        x, y,
        s=20,
        facecolors="none",
        edgecolors="black",
        linewidths=0.5,
    )
    # 2) Fill with color only for non-zero values
    mask = values != 0
    if np.any(mask):
        sc = ax.scatter(
            x[mask],
            y[mask],
            c=values[mask],
            s=20,
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.3,
        )
        return sc
    # All zeros: return a dummy scatter for colorbar (use full range)
    sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
    return sc


def plot_event_2x2(
    x_bottom: np.ndarray,
    y_bottom: np.ndarray,
    x_top: np.ndarray,
    y_top: np.ndarray,
    time_bottom: np.ndarray,
    charge_bottom: np.ndarray,
    time_top: np.ndarray,
    charge_top: np.ndarray,
    z_bottom: float,
    z_top: float,
    event_idx: int,
    label: np.ndarray,
    output_path: Path,
) -> None:
    """
    Draw 4 scatter plots (2x2: bottom/top x time/charge) in one figure and save.
    label: [mccoreX, mccoreY, mczenithAngle, mcazimuthAngle, mclogEnergy].
    Colorbar height matches each subplot rectangle.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Label with units (angle_convergence=False)
    cx, cy, zenith, azimuth, logE = label[0], label[1], label[2], label[3], label[4]
    label_str = (
        f"mccoreX={cx:.3f} m, mccoreY={cy:.3f} m | "
        f"zenith={zenith:.4f} rad, azimuth={azimuth:.4f} rad | "
        f"mclogEnergy={logE:.4f}"
    )

    # Shared norm for time and charge so colorbars are comparable (0 = no fill)
    time_all = np.concatenate([time_bottom, time_top])
    charge_all = np.concatenate([charge_bottom, charge_top])
    norm_time = mcolors.Normalize(vmin=0, vmax=max(time_all.max(), 1e-9))
    norm_charge = mcolors.Normalize(vmin=0, vmax=max(charge_all.max(), 1e-9))

    # [0,0] bottom - time  [0,1] bottom - charge
    # [1,0] top - time      [1,1] top - charge
    sc0 = _scatter_tanks(axes[0, 0], x_bottom, y_bottom, time_bottom, norm=norm_time, cmap=CMAP_TIME)
    axes[0, 0].set_aspect("equal", "box")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    _add_colorbar_same_height(axes[0, 0], sc0, "time [ns]")
    axes[0, 0].set_title(f"Bottom PMTs | time | z={z_bottom:.2f} m")

    sc1 = _scatter_tanks(axes[0, 1], x_bottom, y_bottom, charge_bottom, norm=norm_charge, cmap=CMAP_CHARGE)
    axes[0, 1].set_aspect("equal", "box")
    axes[0, 1].set_xlabel("x [m]")
    axes[0, 1].set_ylabel("y [m]")
    _add_colorbar_same_height(axes[0, 1], sc1, "charge [PE]")
    axes[0, 1].set_title(f"Bottom PMTs | charge | z={z_bottom:.2f} m")

    sc2 = _scatter_tanks(axes[1, 0], x_top, y_top, time_top, norm=norm_time, cmap=CMAP_TIME)
    axes[1, 0].set_aspect("equal", "box")
    axes[1, 0].set_xlabel("x [m]")
    axes[1, 0].set_ylabel("y [m]")
    _add_colorbar_same_height(axes[1, 0], sc2, "time [ns]")
    axes[1, 0].set_title(f"Top PMTs | time | z={z_top:.2f} m")

    sc3 = _scatter_tanks(axes[1, 1], x_top, y_top, charge_top, norm=norm_charge, cmap=CMAP_CHARGE)
    axes[1, 1].set_aspect("equal", "box")
    axes[1, 1].set_xlabel("x [m]")
    axes[1, 1].set_ylabel("y [m]")
    _add_colorbar_same_height(axes[1, 1], sc3, "charge [PE]")
    axes[1, 1].set_title(f"Top PMTs | charge | z={z_top:.2f} m")

    fig.suptitle(f"Event {event_idx}  |  {label_str}", fontsize=10)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a single SWGO event as 4 scatter plots (2x2): "
            "bottom/top PMTs x time/charge."
        )
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path to the HDF5 file.",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        required=True,
        help="Event index to visualize.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default=".",
        help="Directory to save the output PNG (default: current directory).",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default=None,
        help="Output filename (default: event_{index}.png).",
    )
    parser.add_argument(
        "--position-path",
        default="/store/hawc/swgo/ml/M7/position_D9.npz",
        help="Path to NPZ with PMT positions (xpmts, ypmts, zpmts), same as train.",
    )

    args = parser.parse_args()

    h5_path = args.path
    event_idx = args.index

    x, y, z, time, charge, label = load_event(
        h5_path, event_idx, position_path=args.position_path
    )

    (bottom_mask, z_bottom), (top_mask, z_top) = split_top_bottom_by_z(z)

    x_bottom = x[bottom_mask]
    y_bottom = y[bottom_mask]
    x_top = x[top_mask]
    y_top = y[top_mask]

    time_bottom = time[bottom_mask]
    time_top = time[top_mask]
    charge_bottom = charge[bottom_mask]
    charge_top = charge[top_mask]

    out_name = args.out_name if args.out_name is not None else f"event_{event_idx}.png"
    output_path = Path(args.out_dir) / out_name

    plot_event_2x2(
        x_bottom,
        y_bottom,
        x_top,
        y_top,
        time_bottom,
        charge_bottom,
        time_top,
        charge_top,
        z_bottom,
        z_top,
        event_idx,
        label,
        output_path,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

