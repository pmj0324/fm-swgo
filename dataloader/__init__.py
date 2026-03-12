"""
Data loading utilities for SWGO HDF5 files.

Currently exposes:
- H5EventDataset: PyTorch Dataset that reads per-event PMT features and labels.
"""

from .h5_loader import H5EventDataset  # noqa: F401

