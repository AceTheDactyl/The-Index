#!/usr/bin/env python3

# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Research/experimental code
# Severity: MEDIUM RISK
# Risk Types: ['experimental', 'needs_validation']
# File: systems/Ace-Systems/docs/Research/RHZ Stylus/RHZ Stylus Firmware/host/psd_quicklook.py

import sys
import numpy as np
import pandas as pd
from scipy.signal import welch

"""
Quick PSD check of ADS1220 stream grouped per Zipper stage.
Usage: python host/psd_quicklook.py rhz_host_<timestamp>.csv
"""

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
df = df.dropna(subset=["ads_raw", "t_ms"])

t = df["t_ms"].values * 1e-3
fs = 1.0 / np.median(np.diff(t))
print(f"Estimated fs = {fs:.1f} Hz")

for stage_name, group in df.groupby("stage"):
    samples = group["ads_raw"].values
    if len(samples) < 512:
        continue
    freqs, psd = welch(samples - np.mean(samples), fs=fs, nperseg=1024, scaling="density")
    print(f"\nStage {stage_name}:")
    for target in [222, 333, 444, 555, 666, 777]:
        idx = np.argmin(np.abs(freqs - target))
        print(f"  {target:3d} Hz -> PSD={psd[idx]:.3e}")
