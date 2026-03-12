import argparse
import bisect
from typing import Callable, List, Optional, Sequence, Tuple, Union, Dict, Any

import hdf5plugin  # import 만으로 HDF5 압축 플러그인 등록
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5EventDataset(Dataset):
    """
    PyTorch Dataset that reads SWGO HDF5 files event-by-event.

    Definition of one event:
      - Uses all PMTs (722) for:
          * time      : data/time
          * charge    : data/charge
          * xpmt,ypmt,zpmt : data/xpmt, data/ypmt, data/zpmt
      - Stacks them into a feature tensor of shape (722, 5):
          [time, charge, xpmt, ypmt, zpmt]
      - Label is a length-5 vector. If angle_convergence=True (default):
          [mccoreX, mccoreY, u_x, u_y, mclogEnergy]
        with u_x, u_y from mczenithAngle/mcazimuthAngle. If False:
          [mccoreX, mccoreY, mczenithAngle, mcazimuthAngle, mclogEnergy]

    The dataset can take one or multiple HDF5 files and presents them
    as a single contiguous set of events.

    Optional cut: eventnHit >= cut_nhit, mccoreR < cut_mccorer (None = no cut).
    When cut is set, only events passing both conditions are included.
    """

    def __init__(
        self,
        file_paths: Union[str, Sequence[str]],
        position_path: Optional[str] = None,
        angle_convergence: bool = True,
        cut_nhit: Optional[float] = None,
        cut_mccorer: Optional[float] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        self.file_paths: List[str] = list(file_paths)
        self.position_path = position_path
        self.angle_convergence = angle_convergence
        self.cut_nhit = cut_nhit
        self.cut_mccorer = cut_mccorer
        self.transform = transform
        self.target_transform = target_transform

        # Discover number of events in each file from data/charge
        self._file_sizes: List[int] = []
        for path in self.file_paths:
            with h5py.File(path, "r") as f:
                if "data/charge" not in f:
                    raise KeyError(f"'data/charge' dataset not found in file: {path}")
                n_events = f["data/charge"].shape[0]
                self._file_sizes.append(int(n_events))

        if not self._file_sizes:
            raise ValueError("No HDF5 files provided to H5EventDataset.")

        self._cumulative_sizes: List[int] = np.cumsum(self._file_sizes).tolist()

        # If cut is set, build list of (file_idx, local_idx) for events passing the cut
        self._passing_indices: Optional[List[Tuple[int, int]]] = None
        if cut_nhit is not None or cut_mccorer is not None:
            passing: List[Tuple[int, int]] = []
            for file_idx, path in enumerate(self.file_paths):
                with h5py.File(path, "r") as f:
                    data = f["data/data"][:]
                ev_mask = np.ones(len(data), dtype=bool)
                if cut_nhit is not None:
                    ev_mask &= np.asarray(data["eventnHit"]) >= cut_nhit
                if cut_mccorer is not None:
                    ev_mask &= np.asarray(data["mccoreR"]) < cut_mccorer
                for local_idx in np.where(ev_mask)[0]:
                    passing.append((file_idx, int(local_idx)))
            self._passing_indices = passing

        # Optional fixed PMT positions loaded from an npz file
        self._xpmts: Optional[np.ndarray] = None
        self._ypmts: Optional[np.ndarray] = None
        self._zpmts: Optional[np.ndarray] = None

        if self.position_path is not None:
            pos = np.load(self.position_path)
            try:
                xpmts = np.asarray(pos["xpmts"], dtype=np.float32)
                ypmts = np.asarray(pos["ypmts"], dtype=np.float32)
                zpmts = np.asarray(pos["zpmts"], dtype=np.float32)
            finally:
                pos.close()

            if not (xpmts.shape == ypmts.shape == zpmts.shape):
                raise ValueError(
                    f"Position arrays in {self.position_path} must have same shape, "
                    f"got {xpmts.shape}, {ypmts.shape}, {zpmts.shape}"
                )

            self._xpmts = xpmts
            self._ypmts = ypmts
            self._zpmts = zpmts

    def __len__(self) -> int:
        if self._passing_indices is not None:
            return len(self._passing_indices)
        return self._cumulative_sizes[-1]

    def _get_file_index(self, idx: int) -> Tuple[int, int]:
        """Map global index -> (file_idx, local_idx)."""
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        if self._passing_indices is not None:
            return self._passing_indices[idx]
        file_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self._cumulative_sizes[file_idx - 1]
        return file_idx, local_idx

    def _read_event_from_file(self, path: str, local_idx: int) -> Dict[str, Any]:
        """
        Read a single event (features + label) from a given file.
        """
        with h5py.File(path, "r") as f:
            charge = f["data/charge"][local_idx]  # (722,)
            time = f["data/time"][local_idx]      # (722,)

            # Use fixed positions from npz if available, otherwise fall back to HDF5
            if self._xpmts is not None:
                xpmt = self._xpmts
                ypmt = self._ypmts
                zpmt = self._zpmts
            else:
                xpmt = f["data/xpmt"][local_idx]
                ypmt = f["data/ypmt"][local_idx]
                zpmt = f["data/zpmt"][local_idx]

            # Stack into (722, 5): [time, charge, xpmt, ypmt, zpmt]
            features_np = np.stack(
                [time, charge, xpmt, ypmt, zpmt],
                axis=-1,
                dtype=np.float32,
            )

            # Structured array row for labels (length 5)
            row = f["data/data"][local_idx]
            core_x = float(row["mccoreX"])
            core_y = float(row["mccoreY"])
            theta = float(row["mczenithAngle"])
            phi = float(row["mcazimuthAngle"])
            energy = float(row["mclogEnergy"])

            if self.angle_convergence:
                # [mccoreX, mccoreY, u_x, u_y, mclogEnergy]
                ux = np.sin(theta) * np.cos(phi)
                uy = np.sin(theta) * np.sin(phi)
                label_np = np.array([core_x, core_y, ux, uy, energy], dtype=np.float32)
            else:
                # [mccoreX, mccoreY, mczenithAngle, mcazimuthAngle, mclogEnergy]
                label_np = np.array([core_x, core_y, theta, phi, energy], dtype=np.float32)

        return {
            "features": features_np,
            "label": label_np,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, local_idx = self._get_file_index(idx)
        path = self.file_paths[file_idx]

        sample = self._read_event_from_file(path, local_idx)
        x = torch.from_numpy(sample["features"])  # (722, 5)
        y = torch.from_numpy(sample["label"])     # (5,)

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return {"x": x, "y": y}


def _print_one_event(
    h5_path: str,
    index: int = 0,
    position_path: Optional[str] = None,
    angle_convergence: bool = True,
) -> None:
    """
    Utility to quickly inspect one event from the CLI.
    """
    ds = H5EventDataset(
        h5_path,
        position_path=position_path,
        angle_convergence=angle_convergence,
    )
    sample = ds[index]

    x = sample["x"]
    y = sample["y"]

    print(f"HDF5 file: {h5_path}")
    print(f"Event index: {index}")
    print(f"Features shape (n_pmts, n_channels): {tuple(x.shape)}")
    print(f"Label shape: {tuple(y.shape)}")
    if ds.angle_convergence:
        print("Label values [mccoreX, mccoreY, u_x, u_y, mclogEnergy]:")
        print(y)
    else:
        print("Label values [mccoreX, mccoreY, mczenithAngle, mcazimuthAngle, mclogEnergy]:")
        print(y)

    # Print all PMT feature rows so that the full event can be inspected.
    # Use NumPy printing options to avoid "..." truncation.
    print("\nAll PMT feature rows (time, charge, xpmt, ypmt, zpmt):")
    np.set_printoptions(threshold=np.inf, linewidth=200)
    print(x.numpy())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one event from a SWGO HDF5 file using H5EventDataset."
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path to the HDF5 file",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=0,
        help="Event index to print (default: 0)",
    )
    parser.add_argument(
        "--position-path",
        type=str,
        default="/store/hawc/swgo/ml/M7/position_D9.npz",
        help=(
            "Path to NPZ file containing fixed PMT positions (xpmts, ypmts, zpmts). "
            "If empty, positions will be read from the HDF5 file instead."
        ),
    )
    parser.add_argument(
        "--no-angle-convergence",
        action="store_true",
        help=(
            "If set, keep mczenithAngle and mcazimuthAngle as labels instead of "
            "converting them to (u_x, u_y)."
        ),
    )
    args = parser.parse_args()

    pos_path = args.position_path if args.position_path else None
    angle_conv = not args.no_angle_convergence

    _print_one_event(
        args.path,
        args.index,
        position_path=pos_path,
        angle_convergence=angle_conv,
    )


if __name__ == "__main__":
    main()

