"""
Train script for SWGO flow model.
For now only builds the dataloader and runs a short data-loading sanity check.
Adjust the variables below and run: python train.py
"""
import sys

import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Input: edit these and run the script
# -----------------------------------------------------------------------------
DATA_PATHS = [
    "/store/hawc/swgo/ml/M7/gammaD9v40_trainmoreUHE2.hdf5",
]
POSITION_PATH = "/store/hawc/swgo/ml/M7/position_D9.npz"
BATCH_SIZE = 32
NUM_WORKERS = 0
ANGLE_CONVERGENCE = True  # True: labels [mccoreX, mccoreY, u_x, u_y, mclogEnergy]; False: raw angles
# Cut: eventnHit >= CUT_NHIT, mccoreR < CUT_MCCORER only (None = no cut)
CUT_NHIT = None
CUT_MCCORER = None
# -----------------------------------------------------------------------------

sys.path.insert(0, str(__file__.rsplit("/", 1)[0] or "."))
from dataloader.h5_loader import H5EventDataset

dataset = H5EventDataset(
    file_paths=DATA_PATHS,
    position_path=POSITION_PATH,
    angle_convergence=ANGLE_CONVERGENCE,
    cut_nhit=CUT_NHIT,
    cut_mccorer=CUT_MCCORER,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

print(f"Dataset size: {len(dataset)}")
print(f"Batches per epoch: {len(loader)}")
print("Loading a few batches ...")

for i, batch in enumerate(loader):
    x = batch["x"]   # (B, 722, 5)
    y = batch["y"]   # (B, 5)
    print(f"  batch {i}: x {tuple(x.shape)}, y {tuple(y.shape)}")
    if i >= 2:
        break

print("Done (dataloader only).")
