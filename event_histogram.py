"""
Histograms of time and NPE (charge) over events.
Draws two figures: one including zeros, one excluding zeros.
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
POSITION_PATH = "/store/hawc/swgo/ml/M7/position_D9.npz"
MAX_EVENTS = None   # None = all events; set e.g. 50000 to limit
Z_FILTER = "all"    # "all" | "top" | "bottom" | "both"
OUT_DIR = "."
# Cut: eventnHit >= CUT_NHIT; mccoreR < CUT_MCCORER only (None = no cut)
CUT_NHIT = 15
CUT_MCCORER = 120
# -----------------------------------------------------------------------------


def _load_z_from_npz(path: str) -> np.ndarray:
    """Load PMT z positions (722,) from npz. Same convention as plot_event: 0.06=top, 0.07=bottom."""
    with np.load(path) as pos:
        return np.asarray(pos["zpmts"], dtype=np.float32)


def _mask_by_z(z: np.ndarray, z_filter: str):
    """Return boolean mask(s) for PMTs by z. z_filter: all | top | bottom | both."""
    if z_filter == "all":
        return None
    unique_z = np.unique(z)
    if unique_z.size != 2:
        raise ValueError(f"Expected 2 z-levels, got {unique_z.size}")
    z_lo, z_hi = np.sort(unique_z)
    top = z == z_lo
    bottom = z == z_hi
    if z_filter == "top":
        return top
    if z_filter == "bottom":
        return bottom
    if z_filter == "both":
        return (top, bottom)
    raise ValueError(f"Z_FILTER must be all|top|bottom|both, got {z_filter!r}")


def collect_time_charge_bulk(
    data_paths: list,
    position_path: str,
    max_events: int | None,
    z_filter: str,
    cut_nhit: float | None,
    cut_mccorer: float | None,
):
    """
    Bulk-load time and charge from HDF5. Apply cut (eventnHit, mccoreR), then z filter.
    Returns dict with flattened arrays for histogramming, and n_events_after_cut.
    """
    z_pmt = _load_z_from_npz(position_path)  # (722,)
    mask_z = _mask_by_z(z_pmt, z_filter)

    time_chunks = []
    charge_chunks = []
    if z_filter == "both":
        time_top_chunks, time_bot_chunks = [], []
        charge_top_chunks, charge_bot_chunks = [], []
    n_read = 0
    n_after_cut = 0
    for path in data_paths:
        with h5py.File(path, "r") as f:
            time_ds = f["data/time"]
            charge_ds = f["data/charge"]
            data_ds = f["data/data"]
            n_file = time_ds.shape[0]
            if max_events is not None:
                need = max_events - n_read
                if need <= 0:
                    break
                n_slice = min(n_file, need)
            else:
                n_slice = n_file
            time_batch = time_ds[:n_slice]   # (n_slice, 722)
            charge_batch = charge_ds[:n_slice]
            data_batch = data_ds[:n_slice]
        n_read += n_slice
        # Event cut: eventnHit >= cut_nhit; mccoreR < cut_mccorer only
        ev_mask = np.ones(n_slice, dtype=bool)
        if cut_nhit is not None:
            ev_mask &= np.asarray(data_batch["eventnHit"]) >= cut_nhit
        if cut_mccorer is not None:
            ev_mask &= np.asarray(data_batch["mccoreR"]) < cut_mccorer
        time_batch = time_batch[ev_mask]   # (n_pass, 722)
        charge_batch = charge_batch[ev_mask]
        n_after_cut += ev_mask.sum()

        if mask_z is None:
            time_chunks.append(time_batch.ravel())
            charge_chunks.append(charge_batch.ravel())
        elif z_filter == "both":
            top, bottom = mask_z
            time_top_chunks.append(time_batch[:, top].ravel())
            time_bot_chunks.append(time_batch[:, bottom].ravel())
            charge_top_chunks.append(charge_batch[:, top].ravel())
            charge_bot_chunks.append(charge_batch[:, bottom].ravel())
        else:
            time_chunks.append(time_batch[:, mask_z].ravel())
            charge_chunks.append(charge_batch[:, mask_z].ravel())

        if max_events is not None and n_read >= max_events:
            break

    if z_filter == "both":
        return {
            "time_top": np.concatenate(time_top_chunks),
            "time_bottom": np.concatenate(time_bot_chunks),
            "charge_top": np.concatenate(charge_top_chunks),
            "charge_bottom": np.concatenate(charge_bot_chunks),
        }, n_after_cut
    return {
        "time": np.concatenate(time_chunks),
        "charge": np.concatenate(charge_chunks),
    }, n_after_cut


def _plot_one_figure(
    data,
    z_filter: str,
    include_zeros: bool,
    out_path: Path,
    n_events: int,
    log_scale: bool = False,
):
    """Draw one figure (time + charge histograms) and save.
    If log_scale=True, x and y axes are log; only positive time/charge are used for log x.
    """
    fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(10, 4))

    if log_scale:
        # For log x we need strictly positive values
        def _pos(x):
            return x[x > 0]

    if z_filter == "both":
        if include_zeros and not log_scale:
            ax_t.hist(data["time_top"], bins=80, alpha=0.6, label="top", color="C0", density=True)
            ax_t.hist(data["time_bottom"], bins=80, alpha=0.6, label="bottom", color="C1", density=True)
            ax_c.hist(data["charge_top"], bins=80, alpha=0.6, label="top", color="C0", density=True)
            ax_c.hist(data["charge_bottom"], bins=80, alpha=0.6, label="bottom", color="C1", density=True)
        else:
            if log_scale:
                tt, tb = _pos(data["time_top"]), _pos(data["time_bottom"])
                ct, cb = _pos(data["charge_top"]), _pos(data["charge_bottom"])
            else:
                tt = data["time_top"][data["time_top"] != 0]
                tb = data["time_bottom"][data["time_bottom"] != 0]
                ct = data["charge_top"][data["charge_top"] != 0]
                cb = data["charge_bottom"][data["charge_bottom"] != 0]
            if len(tt): ax_t.hist(tt, bins=80, alpha=0.6, label="top", color="C0", density=True)
            if len(tb): ax_t.hist(tb, bins=80, alpha=0.6, label="bottom", color="C1", density=True)
            if len(ct): ax_c.hist(ct, bins=80, alpha=0.6, label="top", color="C0", density=True)
            if len(cb): ax_c.hist(cb, bins=80, alpha=0.6, label="bottom", color="C1", density=True)
        ax_t.legend()
        ax_c.legend()
    else:
        t = data["time"]
        c = data["charge"]
        if not include_zeros or log_scale:
            if log_scale:
                t, c = _pos(t), _pos(c)
            else:
                t, c = t[t != 0], c[c != 0]
        if len(t):
            ax_t.hist(t, bins=80, color="steelblue", alpha=0.8, edgecolor="white", density=True)
        if len(c):
            ax_c.hist(c, bins=80, color="steelblue", alpha=0.8, edgecolor="white", density=True)

    if log_scale:
        ax_t.set_xscale("log")
        ax_t.set_yscale("log")
        ax_c.set_xscale("log")
        ax_c.set_yscale("log")
        ax_t.set_ylim(bottom=1e-8)
        ax_c.set_ylim(bottom=1e-8)

    ax_t.set_xlabel("time [ns]")
    ax_t.set_ylabel("density")
    ax_t.set_title("time")
    ax_c.set_xlabel("charge [PE]")
    ax_c.set_ylabel("density")
    ax_c.set_title("NPE (charge)")

    title = "with zeros" if include_zeros else "excluding zeros"
    if log_scale:
        title += " (log x, log y)"
    fig.suptitle(f"Event time & NPE ({title}, n_events={n_events}, z={Z_FILTER})", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cut_str = f" cut=(eventnHit>={CUT_NHIT}, mccoreR<{CUT_MCCORER})" if (CUT_NHIT is not None or CUT_MCCORER is not None) else ""
    print(f"Loading events (z={Z_FILTER}){cut_str}...")
    data, n_events = collect_time_charge_bulk(
        DATA_PATHS, POSITION_PATH, MAX_EVENTS, Z_FILTER, CUT_NHIT, CUT_MCCORER
    )
    print(f"Loaded {n_events} events (after cut).")

    out_dir = Path(OUT_DIR)
    _plot_one_figure(
        data, Z_FILTER, include_zeros=True,
        out_path=out_dir / "event_histogram_with_zeros.png",
        n_events=n_events,
    )
    _plot_one_figure(
        data, Z_FILTER, include_zeros=False,
        out_path=out_dir / "event_histogram_no_zeros.png",
        n_events=n_events,
    )
    _plot_one_figure(
        data, Z_FILTER, include_zeros=True,
        out_path=out_dir / "event_histogram_with_zeros_log.png",
        n_events=n_events,
        log_scale=True,
    )
    _plot_one_figure(
        data, Z_FILTER, include_zeros=False,
        out_path=out_dir / "event_histogram_no_zeros_log.png",
        n_events=n_events,
        log_scale=True,
    )
    print(f"Saved: {out_dir / 'event_histogram_with_zeros.png'}")
    print(f"Saved: {out_dir / 'event_histogram_no_zeros.png'}")
    print(f"Saved: {out_dir / 'event_histogram_with_zeros_log.png'}")
    print(f"Saved: {out_dir / 'event_histogram_no_zeros_log.png'}")


if __name__ == "__main__":
    main()
