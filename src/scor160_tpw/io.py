"""I/O for ATOMIX-format NetCDF benchmark files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import netCDF4
import numpy as np


@dataclass
class L1Data:
    """Level-1 converted time series in physical units."""

    time: np.ndarray  # (N_TIME,), decimal days
    pres: np.ndarray  # (N_TIME,), dbar
    shear: np.ndarray  # (N_SHEAR, N_TIME), s⁻¹
    # Vibration/acceleration — whichever the instrument provides
    vib: np.ndarray  # (N_VIB, N_TIME)
    vib_type: str  # "ACC" or "VIB"

    fs_fast: float  # fast sampling rate [Hz]
    f_AA: float  # anti-alias filter cutoff [Hz]
    vehicle: str  # "vmp", "mss", etc.
    profile_dir: str  # "down", "up", or "horizontal"
    time_reference_year: int

    # Optional: pre-computed speed (e.g. Epsifish, MSS provide this in L1)
    pspd_rel: np.ndarray = field(default_factory=lambda: np.array([]))

    # Optional slow-rate arrays (may be empty)
    time_slow: np.ndarray = field(default_factory=lambda: np.array([]))
    pres_slow: np.ndarray = field(default_factory=lambda: np.array([]))
    pitch: np.ndarray = field(default_factory=lambda: np.array([]))
    roll: np.ndarray = field(default_factory=lambda: np.array([]))
    temp: np.ndarray = field(default_factory=lambda: np.array([]))
    fs_slow: float = 0.0

    @property
    def n_shear(self) -> int:
        return self.shear.shape[0]

    @property
    def n_vib(self) -> int:
        return self.vib.shape[0]

    @property
    def n_time(self) -> int:
        return self.time.shape[0]

    @property
    def has_speed(self) -> bool:
        return self.pspd_rel.size > 0


@dataclass
class L2Params:
    """Processing parameters for L1→L2, read from L2_cleaned group attributes."""

    HP_cut: float  # high-pass cutoff frequency [Hz]
    despike_sh: np.ndarray  # [threshold, smooth_freq, half_width_frac]
    despike_A: np.ndarray  # same for accelerometers
    profile_min_W: float  # minimum speed [m/s]
    profile_min_P: float  # minimum pressure [dbar]
    profile_min_duration: float  # minimum section duration [s]


@dataclass
class L2Data:
    """Level-2 cleaned time series."""

    time: np.ndarray
    shear: np.ndarray  # (N_SHEAR, N_TIME), despiked + HP-filtered
    vib: np.ndarray  # (N_VIB, N_TIME), despiked + HP-filtered
    vib_type: str
    pspd_rel: np.ndarray  # (N_TIME,), profiling speed [m/s]
    section_number: np.ndarray  # (N_TIME,), 0 = excluded


@dataclass
class L3Params:
    """Processing parameters for L2→L3, read from L3_spectra group attributes."""

    fft_length: int  # FFT segment length [samples]
    diss_length: int  # dissipation window length [samples]
    overlap: int  # dissipation window overlap [samples]
    HP_cut: float  # high-pass cutoff frequency [Hz]
    fs_fast: float  # fast sampling rate [Hz]
    goodman: bool  # whether Goodman cleaning was applied


@dataclass
class L3Data:
    """Level-3 wavenumber spectra."""

    time: np.ndarray  # (N_SPECTRA,), center time of each window
    pres: np.ndarray  # (N_SPECTRA,), mean pressure per window
    temp: np.ndarray  # (N_SPECTRA,), mean temperature per window
    pspd_rel: np.ndarray  # (N_SPECTRA,), mean speed per window
    section_number: np.ndarray  # (N_SPECTRA,)
    kcyc: np.ndarray  # (N_WAVENUMBER, N_SPECTRA), wavenumber [cpm]
    sh_spec: np.ndarray  # (N_SHEAR, N_WAVENUMBER, N_SPECTRA), raw spectrum
    sh_spec_clean: np.ndarray  # (N_SHEAR, N_WAVENUMBER, N_SPECTRA), Goodman-cleaned

    @property
    def n_spectra(self) -> int:
        return self.time.shape[0]

    @property
    def n_wavenumber(self) -> int:
        return self.kcyc.shape[0]

    @property
    def n_shear(self) -> int:
        return self.sh_spec.shape[0]


@dataclass
class L4Params:
    """Processing parameters for L3→L4, read from L4_dissipation group attributes."""

    fft_length: int  # FFT segment length [samples]
    diss_length: int  # dissipation window length [samples]
    overlap: int  # dissipation window overlap [samples]
    fs_fast: float  # fast sampling rate [Hz]
    fit_order: int  # polynomial order for spectral minimum detection
    f_AA: float  # anti-alias filter cutoff [Hz]
    FOM_limit: float  # QC threshold for figure of merit
    variance_resolved_limit: float  # QC threshold for variance resolved


@dataclass
class L4Data:
    """Level-4 dissipation estimates."""

    time: np.ndarray  # (N_SPECTRA,)
    pres: np.ndarray  # (N_SPECTRA,)
    pspd_rel: np.ndarray  # (N_SPECTRA,)
    section_number: np.ndarray  # (N_SPECTRA,)
    epsi: np.ndarray  # (N_SHEAR, N_SPECTRA), epsilon per probe
    epsi_final: np.ndarray  # (N_SPECTRA,), combined epsilon
    epsi_flags: np.ndarray  # (N_SHEAR, N_SPECTRA), QC flags
    fom: np.ndarray  # (N_SHEAR, N_SPECTRA), figure of merit
    mad: np.ndarray  # (N_SHEAR, N_SPECTRA), mean absolute deviation
    kmax: np.ndarray  # (N_SHEAR, N_SPECTRA), upper integration limit
    method: np.ndarray  # (N_SHEAR, N_SPECTRA), 0=variance, 1=ISR
    var_resolved: np.ndarray  # (N_SHEAR, N_SPECTRA), fraction of variance resolved

    @property
    def n_spectra(self) -> int:
        return self.time.shape[0]

    @property
    def n_shear(self) -> int:
        return self.epsi.shape[0]


def read_atomix(path: str | Path) -> tuple[L1Data, L2Params, L2Data, L3Params, L3Data, L4Data]:
    """Read an ATOMIX benchmark NetCDF file.

    Returns L1 data, L2 processing parameters, reference L2 result,
    L3 processing parameters, reference L3 result, and reference L4 result.
    """
    path = Path(path)
    ds = netCDF4.Dataset(str(path), "r")
    try:
        l1 = _read_l1(ds)
        l2_params = _read_l2_params(ds, l1)
        l2_ref = _read_l2(ds)
        l3_params = _read_l3_params(ds)
        l3_ref = _read_l3(ds)
        l4_ref = _read_l4(ds)

        # If L1 has no speed but L2 does (e.g. Nemo: speed from hotel file),
        # copy L2 reference speed into L1 so process_l2 can use it.
        if not l1.has_speed and l2_ref.pspd_rel.shape[0] == l1.n_time:
            l1.pspd_rel = l2_ref.pspd_rel.copy()
    finally:
        ds.close()
    return l1, l2_params, l2_ref, l3_params, l3_ref, l4_ref


# ---------------------------------------------------------------------------
# Helpers for resolving attributes across groups
# ---------------------------------------------------------------------------

def _resolve_attr(ds: netCDF4.Dataset, name: str, default=None):
    """Look for an attribute in L1_converted, then L2_cleaned, then root."""
    for source in [ds.groups["L1_converted"], ds.groups["L2_cleaned"], ds]:
        if name in {a for a in source.ncattrs()}:
            return source.getncattr(name)
    return default


# ---------------------------------------------------------------------------
# L1 reader
# ---------------------------------------------------------------------------

def _read_l1(ds: netCDF4.Dataset) -> L1Data:
    g = ds.groups["L1_converted"]

    time = np.asarray(g.variables["TIME"][:], dtype=np.float64)
    shear = np.asarray(g.variables["SHEAR"][:], dtype=np.float64)

    # Pressure may be on a slower grid (e.g. Epsifish: TIME_CTD)
    pres_var = g.variables["PRES"]
    pres_raw = np.asarray(pres_var[:], dtype=np.float64)
    if pres_raw.shape[0] == len(time):
        pres = pres_raw
    else:
        pres_dim = pres_var.dimensions[0]
        if pres_dim in g.variables:
            time_pres = np.asarray(g.variables[pres_dim][:], dtype=np.float64)
            pres = np.interp(time, time_pres, pres_raw)
        else:
            pres = np.interp(
                np.linspace(0, 1, len(time)),
                np.linspace(0, 1, len(pres_raw)),
                pres_raw,
            )

    # ACC or VIB
    if "ACC" in g.variables:
        vib = np.asarray(g.variables["ACC"][:], dtype=np.float64)
        vib_type = "ACC"
    elif "VIB" in g.variables:
        vib = np.asarray(g.variables["VIB"][:], dtype=np.float64)
        vib_type = "VIB"
    else:
        vib = np.zeros((0, len(time)), dtype=np.float64)
        vib_type = "NONE"

    # Pre-computed speed (Epsifish, MSS provide this in L1).
    # May be on a slower time grid (e.g. CTD rate) — interpolate to fast.
    pspd_rel = np.array([])
    if "PSPD_REL" in g.variables:
        pspd_var = g.variables["PSPD_REL"]
        pspd_raw = np.asarray(pspd_var[:], dtype=np.float64)
        if pspd_raw.shape[0] == len(time):
            pspd_rel = pspd_raw
        else:
            # On a different time grid — find the corresponding time variable
            pspd_dims = pspd_var.dimensions
            time_dim = pspd_dims[0]  # e.g. "TIME_CTD"
            if time_dim in g.variables:
                time_slow_spd = np.asarray(g.variables[time_dim][:], dtype=np.float64)
                pspd_rel = np.interp(time, time_slow_spd, pspd_raw)
            else:
                # Fallback: assume uniform spacing, just resample
                pspd_rel = np.interp(
                    np.linspace(0, 1, len(time)),
                    np.linspace(0, 1, len(pspd_raw)),
                    pspd_raw,
                )

    # Optional slow-rate channels
    time_slow = np.array([])
    pres_slow = np.array([])
    pitch = np.array([])
    roll = np.array([])
    temp = np.array([])

    if "TIME_SLOW" in g.variables:
        time_slow = np.asarray(g.variables["TIME_SLOW"][:], dtype=np.float64)
    if "PRES_SLOW" in g.variables:
        pres_slow = np.asarray(g.variables["PRES_SLOW"][:], dtype=np.float64)
    if "PITCH" in g.variables:
        pitch = np.asarray(g.variables["PITCH"][:], dtype=np.float64)
    if "ROLL" in g.variables:
        roll = np.asarray(g.variables["ROLL"][:], dtype=np.float64)
    if "TEMP" in g.variables:
        temp = np.asarray(g.variables["TEMP"][:], dtype=np.float64)

    fs_fast = float(_resolve_attr(ds, "fs_fast", 512.0))
    fs_slow = float(_resolve_attr(ds, "fs_slow", 0.0))

    return L1Data(
        time=time,
        pres=pres,
        shear=shear,
        vib=vib,
        vib_type=vib_type,
        fs_fast=fs_fast,
        f_AA=float(_resolve_attr(ds, "f_AA", 98.0)),
        vehicle=str(_resolve_attr(ds, "vehicle", "unknown")),
        profile_dir=str(_resolve_attr(ds, "profile_dir", "down")),
        time_reference_year=int(_resolve_attr(ds, "time_reference_year", 2000)),
        pspd_rel=pspd_rel,
        time_slow=time_slow,
        pres_slow=pres_slow,
        pitch=pitch,
        roll=roll,
        temp=temp,
        fs_slow=fs_slow,
    )


# ---------------------------------------------------------------------------
# L2 params reader
# ---------------------------------------------------------------------------

def _read_l2_params(ds: netCDF4.Dataset, l1: L1Data) -> L2Params:
    """Read L2 processing parameters, falling back to sensible defaults."""
    g = ds.groups["L2_cleaned"]
    attrs = {a: g.getncattr(a) for a in g.ncattrs()}

    # Defaults per Lueck et al. (2024) recommendations
    HP_cut = float(attrs.get("HP_cut", 0.25))
    despike_sh = np.asarray(
        attrs.get("despike_sh", [8.0, 0.5, 0.04]), dtype=np.float64
    )
    despike_A = np.asarray(
        attrs.get("despike_A", [np.inf, 0.5, 0.04]), dtype=np.float64
    )
    profile_min_W = float(attrs.get("profile_min_W", 0.1))
    profile_min_P = float(attrs.get("profile_min_P", 1.0))
    profile_min_duration = float(attrs.get("profile_min_duration", 10.0))

    return L2Params(
        HP_cut=HP_cut,
        despike_sh=despike_sh,
        despike_A=despike_A,
        profile_min_W=profile_min_W,
        profile_min_P=profile_min_P,
        profile_min_duration=profile_min_duration,
    )


# ---------------------------------------------------------------------------
# L2 reference reader
# ---------------------------------------------------------------------------

def _read_l2(ds: netCDF4.Dataset) -> L2Data:
    g = ds.groups["L2_cleaned"]
    time = np.asarray(g.variables["TIME"][:], dtype=np.float64)

    shear = np.asarray(g.variables["SHEAR"][:], dtype=np.float64)
    section_number = np.asarray(g.variables["SECTION_NUMBER"][:], dtype=np.float64)

    # PSPD_REL may be on a slower time grid (e.g. Epsifish: TIME_CTD)
    pspd_var = g.variables["PSPD_REL"]
    pspd_raw = np.asarray(pspd_var[:], dtype=np.float64)
    if pspd_raw.shape[0] == len(time):
        pspd_rel = pspd_raw
    else:
        pspd_dim = pspd_var.dimensions[0]
        if pspd_dim in g.variables:
            time_pspd = np.asarray(g.variables[pspd_dim][:], dtype=np.float64)
            pspd_rel = np.interp(time, time_pspd, pspd_raw)
        else:
            pspd_rel = np.interp(
                np.linspace(0, 1, len(time)),
                np.linspace(0, 1, len(pspd_raw)),
                pspd_raw,
            )

    if "ACC" in g.variables:
        vib = np.asarray(g.variables["ACC"][:], dtype=np.float64)
        vib_type = "ACC"
    elif "VIB" in g.variables:
        vib = np.asarray(g.variables["VIB"][:], dtype=np.float64)
        vib_type = "VIB"
    else:
        vib = np.zeros((0, len(time)), dtype=np.float64)
        vib_type = "NONE"

    return L2Data(
        time=time,
        shear=shear,
        vib=vib,
        vib_type=vib_type,
        pspd_rel=pspd_rel,
        section_number=section_number,
    )


# ---------------------------------------------------------------------------
# L3 params reader
# ---------------------------------------------------------------------------

def _read_l3_params(ds: netCDF4.Dataset) -> L3Params:
    """Read L3 processing parameters from L3_spectra group attributes."""
    g = ds.groups["L3_spectra"]
    attrs = {a: g.getncattr(a) for a in g.ncattrs()}

    fs_fast = float(attrs.get("fs_fast", _resolve_attr(ds, "fs_fast", 512.0)))

    return L3Params(
        fft_length=int(attrs.get("fft_length", 512)),
        diss_length=int(attrs.get("diss_length", 2048)),
        overlap=int(attrs.get("overlap", 1024)),
        HP_cut=float(attrs.get("HP_cut", 0.25)),
        fs_fast=fs_fast,
        goodman=bool(attrs.get("goodman", True)),
    )


# ---------------------------------------------------------------------------
# L3 reference reader
# ---------------------------------------------------------------------------

def _read_l3(ds: netCDF4.Dataset) -> L3Data:
    g = ds.groups["L3_spectra"]

    time = np.asarray(g.variables["TIME"][:], dtype=np.float64)
    pres = np.asarray(g.variables["PRES"][:], dtype=np.float64)
    pspd_rel = np.asarray(g.variables["PSPD_REL"][:], dtype=np.float64)
    section_number = np.asarray(g.variables["SECTION_NUMBER"][:], dtype=np.float64)

    temp = np.zeros_like(time)
    if "TEMP" in g.variables:
        temp = np.asarray(g.variables["TEMP"][:], dtype=np.float64)

    kcyc = np.asarray(g.variables["KCYC"][:], dtype=np.float64)
    sh_spec = np.asarray(g.variables["SH_SPEC"][:], dtype=np.float64)

    sh_spec_clean = sh_spec.copy()
    if "SH_SPEC_CLEAN" in g.variables:
        sh_spec_clean = np.asarray(g.variables["SH_SPEC_CLEAN"][:], dtype=np.float64)

    return L3Data(
        time=time,
        pres=pres,
        temp=temp,
        pspd_rel=pspd_rel,
        section_number=section_number,
        kcyc=kcyc,
        sh_spec=sh_spec,
        sh_spec_clean=sh_spec_clean,
    )


# ---------------------------------------------------------------------------
# L4 reference reader
# ---------------------------------------------------------------------------

def _read_l4(ds: netCDF4.Dataset) -> L4Data:
    g = ds.groups["L4_dissipation"]

    time = np.asarray(g.variables["TIME"][:], dtype=np.float64)
    pres = np.asarray(g.variables["PRES"][:], dtype=np.float64)
    pspd_rel = np.asarray(g.variables["PSPD_REL"][:], dtype=np.float64)
    section_number = np.asarray(g.variables["SECTION_NUMBER"][:], dtype=np.float64)

    epsi = np.asarray(g.variables["EPSI"][:], dtype=np.float64)
    epsi_final = np.asarray(g.variables["EPSI_FINAL"][:], dtype=np.float64)
    epsi_flags = np.asarray(g.variables["EPSI_FLAGS"][:], dtype=np.float64)
    fom = np.asarray(g.variables["FOM"][:], dtype=np.float64)
    mad = np.asarray(g.variables["MAD"][:], dtype=np.float64)
    kmax = np.asarray(g.variables["KMAX"][:], dtype=np.float64)
    method = np.asarray(g.variables["METHOD"][:], dtype=np.float64)
    var_resolved = np.asarray(g.variables["VAR_RESOLVED"][:], dtype=np.float64)

    return L4Data(
        time=time,
        pres=pres,
        pspd_rel=pspd_rel,
        section_number=section_number,
        epsi=epsi,
        epsi_final=epsi_final,
        epsi_flags=epsi_flags,
        fom=fom,
        mad=mad,
        kmax=kmax,
        method=method,
        var_resolved=var_resolved,
    )
