"""L2 chi: cleaned temperature and vibration for chi estimation.

Despikes fast temperature, HP-filters vibration for Goodman cleaning,
and computes temperature gradient (dT/dz) from the cleaned signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt

from odas_tpw.scor160.despike import despike
from odas_tpw.scor160.io import L1Data, L2Data


@dataclass
class L2ChiData:
    """Level-2 cleaned data for chi estimation."""

    time: np.ndarray  # (N_TIME,)
    pres: np.ndarray  # (N_TIME,), dbar
    temp: np.ndarray  # (N_TIME,), slow-rate temperature [°C] (for viscosity)
    temp_fast: np.ndarray  # (N_TEMP, N_TIME), despiked temperature [°C]
    gradt: np.ndarray  # (N_TEMP, N_TIME), temperature gradient [°C/m]
    vib: np.ndarray  # (N_VIB, N_TIME), HP-filtered vibration
    pspd_rel: np.ndarray  # (N_TIME,), profiling speed [m/s]
    section_number: np.ndarray  # (N_TIME,), 0 = excluded
    diff_gains: list[float]
    fs_fast: float

    @property
    def n_temp(self) -> int:
        return self.temp_fast.shape[0] if self.temp_fast.ndim == 2 else 0

    @property
    def n_vib(self) -> int:
        return self.vib.shape[0]

    @property
    def n_time(self) -> int:
        return self.time.shape[0]


@dataclass
class L2ChiParams:
    """Processing parameters for L2 chi cleaning."""

    HP_cut: float = 0.25  # HP cutoff for vibration [Hz]
    despike_T: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 0.5, 0.04]),
    )  # [threshold, smooth_freq, half_width_fraction]


def process_l2_chi(
    l1: L1Data,
    l2: L2Data,
    params: L2ChiParams | None = None,
) -> L2ChiData:
    """Clean L1 temperature and vibration for chi estimation.

    Steps:
      1. Despike fast temperature within sections.
      2. HP-filter vibration for Goodman coherent noise removal.
      3. Compute temperature gradient: dT/dz = fs * diff(T) / speed.

    Parameters
    ----------
    l1 : L1Data
        Level-1 data with raw temp_fast and vib.
    l2 : L2Data
        Level-2 data for section_number and pspd_rel.
    params : L2ChiParams, optional
        Cleaning parameters. Uses defaults if None.

    Returns
    -------
    L2ChiData
    """
    if not l1.has_temp_fast:
        raise ValueError("L1Data has no temp_fast — cannot compute chi")

    if params is None:
        params = L2ChiParams()

    fs = l1.fs_fast
    n_temp = l1.n_temp
    diff_gains = l1.diff_gains if l1.diff_gains else [0.94] * n_temp

    # 1. Despike temperature within sections
    temp_fast = np.array(l1.temp_fast, dtype=np.float64)
    t_thresh, t_smooth, t_hw = params.despike_T

    if np.isfinite(t_thresh):
        sections = np.unique(l2.section_number)
        sections = sections[sections > 0]
        for sec_id in sections:
            mask = l2.section_number == sec_id
            for ci in range(n_temp):
                N_t = round(t_hw * fs) if t_hw < 1 else int(t_hw)
                temp_fast[ci, mask], _, _, _ = despike(
                    temp_fast[ci, mask],
                    fs,
                    thresh=t_thresh,
                    smooth=t_smooth,
                    N=N_t,
                )

    # 2. HP-filter vibration for Goodman cleaning
    vib = l1.vib.copy() if l1.vib.size > 0 else l1.vib
    if vib.size > 0 and params.HP_cut > 0:
        b, a = butter(1, params.HP_cut / (fs / 2), btype="high")
        for vi in range(l1.n_vib):
            vib[vi] = filtfilt(b, a, vib[vi])

    # 3. Compute temperature gradient: dT/dz = fs * diff(T) / speed
    spd_safe = np.maximum(np.abs(l2.pspd_rel), 0.01)
    gradt = np.empty_like(temp_fast)
    for ci in range(n_temp):
        dTdt = np.diff(temp_fast[ci]) * fs
        dTdt = np.append(dTdt, dTdt[-1])
        gradt[ci] = dTdt / spd_safe

    # Temperature for viscosity
    if l1.temp.size == l1.n_time:
        temp_slow = l1.temp
    elif n_temp > 0:
        temp_slow = temp_fast[0]
    else:
        temp_slow = np.full(l1.n_time, 10.0)

    return L2ChiData(
        time=l1.time.copy(),
        pres=l1.pres.copy(),
        temp=temp_slow,
        temp_fast=temp_fast,
        gradt=gradt,
        vib=vib,
        pspd_rel=l2.pspd_rel.copy(),
        section_number=l2.section_number.copy(),
        diff_gains=diff_gains,
        fs_fast=fs,
    )
