#!/usr/bin/env python
"""Compare ATOMIX benchmark datasets with rsi-tpw epsilon estimates.

Downloads (or reads pre-downloaded) ATOMIX shear-probe benchmark NetCDF
files, feeds their L3 cleaned spectra through the rsi-tpw epsilon
estimation algorithm, and compares results against the benchmark L4
dissipation estimates.

The benchmark datasets are described in:

    Fer, I., Dengler, M., Holtermann, P., Le Boyer, A. & Lueck, R.
    ATOMIX benchmark datasets for dissipation rate measurements using
    shear probes. Scientific Data 11, 518 (2024).
    https://doi.org/10.1038/s41597-024-03323-y

Code repository:
    https://github.com/SCOR-ATOMIX/ShearProbes_BenchmarkDescriptor

Usage
-----
    # Place benchmark .nc files in benchmarks/atomix/ then:
    python scripts/compare_atomix.py

    # Or specify a directory:
    python scripts/compare_atomix.py --data-dir /path/to/atomix/

    # Download benchmark files (requires internet):
    python scripts/compare_atomix.py --download
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# rsi-tpw imports
from odas_tpw.scor160.l4 import E_ISR_THRESHOLD, _estimate_epsilon
from odas_tpw.scor160.ocean import visc35

# ---------------------------------------------------------------------------
# ATOMIX benchmark dataset catalog
# ---------------------------------------------------------------------------

ATOMIX_DATASETS = {
    "FaroeBank": {
        "name": "Faroe Bank Channel",
        "instrument": "VMP2000",
        "citation": "Fer (2023)",
        "doi": "10.5285/05f21d1d-bf9c-5549-e063-6c86abc0b846",
        "file_pattern": "faroebank",
        "notes": (
            "Deep overflow, VMP2000 SN 9, fs=512 Hz. diss_length=8 s, fft_length=2 s, 50% overlap."
        ),
    },
    "TidalChannel": {
        "name": "Haro Strait (Tidal Channel)",
        "instrument": "VMP250",
        "citation": "Lueck (2024)",
        "doi": "10.5285/0ec16a65-abdf-2822-e063-6c86abc06533",
        "file_pattern": "tidalchannel",
        "exclude_pattern": "_cs",  # exclude constant-speed variant
        "notes": (
            "VMP-250-IR SN 215, fs=512 Hz, f_AA=98 Hz. HP filter 0.4 Hz, despike threshold=8."
        ),
    },
    "TidalChannel_cs": {
        "name": "Haro Strait (constant speed)",
        "instrument": "VMP250",
        "citation": "Lueck (2024)",
        "doi": "10.5285/0ec16a65-abdf-2822-e063-6c86abc06533",
        "file_pattern": "tidalchannel_024_cs",
        "notes": ("VMP-250-IR SN 215, constant W=0.75 m/s. Same raw data as TidalChannel."),
    },
    "RockallTrough": {
        "name": "Rockall Trough",
        "instrument": "Epsilometer",
        "citation": "Le Boyer et al. (2024)",
        "doi": "10.5285/0ebffc86-ed32-5dde-e063-6c86abc08b3a",
        "file_pattern": "epsifish",
        "notes": (
            "Epsilometer, fs=320 Hz. Airfoil shear probes. diss_length=10.4 s, fft_length=2.08 s."
        ),
    },
    "BalticSea": {
        "name": "Baltic Sea (Bornholm Basin)",
        "instrument": "MSS90-L",
        "citation": "Holtermann (2024)",
        "doi": "10.5285/0e35f96f-57e3-540b-e063-6c86abc06660",
        "file_pattern": "baltic",
        "notes": (
            "MSS-Microstructure profiler SN 38, Sea & Sun Technology. "
            "fs=1024 Hz, diss_length=5 s, fft_length=2 s."
        ),
    },
    "MinasPassage": {
        "name": "Minas Passage",
        "instrument": "MR1000",
        "citation": "Lueck & Hay (2024)",
        "doi": "10.5285/0ec17274-7a64-2b28-e063-6c86abc0ee02",
        "file_pattern": "minas",
        "notes": ("Moored horizontal MicroRider-1000, fs=2048 Hz, f_AA=392 Hz. 4 shear probes."),
    },
}

BODC_BASE = "https://doi.org"

# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class SpectrumResult:
    """Result of epsilon estimation from one spectrum."""

    idx: int
    probe: int
    epsilon_atomix: float
    epsilon_rsi: float
    kmin_atomix: float
    kmax_atomix: float
    kmax_rsi: float
    method_atomix: int
    method_rsi: int
    fom_atomix: float
    fom_rsi: float
    mad_rsi: float
    speed: float
    pressure: float
    nu: float


@dataclass
class DatasetResult:
    """Aggregated results for one benchmark dataset."""

    name: str
    key: str
    nc_path: Path
    n_spectra: int
    n_probes: int
    spectra: list[SpectrumResult] = field(default_factory=list)

    # Summary statistics (populated after comparison)
    log10_bias: float = np.nan
    log10_rmsd: float = np.nan
    correlation: float = np.nan
    frac_within_half_decade: float = np.nan
    frac_within_1_decade: float = np.nan

    def compute_summary(self) -> None:
        """Compute summary statistics from individual spectrum results."""
        if not self.spectra:
            return
        eps_atomix = np.array([s.epsilon_atomix for s in self.spectra])
        eps_rsi = np.array([s.epsilon_rsi for s in self.spectra])

        valid = (eps_atomix > 0) & (eps_rsi > 0) & np.isfinite(eps_atomix) & np.isfinite(eps_rsi)
        if valid.sum() < 2:
            return

        log_a = np.log10(eps_atomix[valid])
        log_r = np.log10(eps_rsi[valid])

        self.log10_bias = float(np.mean(log_r - log_a))
        self.log10_rmsd = float(np.sqrt(np.mean((log_r - log_a) ** 2)))
        self.correlation = float(np.corrcoef(log_a, log_r)[0, 1])
        diff = np.abs(log_r - log_a)
        self.frac_within_half_decade = float(np.mean(diff < 0.5))
        self.frac_within_1_decade = float(np.mean(diff < 1.0))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def find_benchmark_files(data_dir: Path) -> dict[str, Path]:
    """Find ATOMIX benchmark NetCDF files in the given directory (recursive)."""
    found = {}
    nc_files = sorted(data_dir.rglob("*.nc"))

    for key, info in ATOMIX_DATASETS.items():
        pattern = info["file_pattern"]
        exclude = info.get("exclude_pattern")
        for nc in nc_files:
            name_lower = nc.name.lower()
            if pattern in name_lower:
                if exclude and exclude.lower() in name_lower:
                    continue
                found[key] = nc
                break

    return found


def print_download_instructions(data_dir: Path, missing: list[str]) -> None:
    """Print instructions for downloading missing benchmark files."""
    print("\n" + "=" * 72)
    print("ATOMIX Benchmark Download Instructions")
    print("=" * 72)
    print(f"\nPlace downloaded .nc files in: {data_dir}/\n")
    for key in missing:
        info = ATOMIX_DATASETS[key]
        print(f"  {info['name']} ({info['instrument']})")
        print(f"    DOI: {BODC_BASE}/{info['doi']}")
        print()
    print(
        "Files are hosted by the British Oceanographic Data Centre (BODC).\n"
        "Visit each DOI link, then download the NetCDF file.\n"
    )


def download_benchmark(data_dir: Path, key: str) -> Path | None:
    """Attempt to download a benchmark file from BODC.

    BODC requires navigating a landing page, so this provides the DOI
    URL and returns None if automatic download isn't possible.
    """
    import webbrowser

    info = ATOMIX_DATASETS[key]
    url = f"{BODC_BASE}/{info['doi']}"
    print(f"  Opening browser for {info['name']}: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        print(f"  Could not open browser. Visit: {url}")
    return None


# ---------------------------------------------------------------------------
# ATOMIX NetCDF readers
# ---------------------------------------------------------------------------


def read_atomix_l3(nc_path: Path) -> xr.Dataset:
    """Read the L3_spectra group from an ATOMIX NetCDF file."""
    return xr.open_dataset(nc_path, group="L3_spectra")


def read_atomix_l4(nc_path: Path) -> xr.Dataset:
    """Read the L4_dissipation group from an ATOMIX NetCDF file."""
    return xr.open_dataset(nc_path, group="L4_dissipation")


def read_atomix_attrs(nc_path: Path) -> dict[str, Any]:
    """Read global attributes from an ATOMIX NetCDF file."""
    with xr.open_dataset(nc_path) as ds:
        return dict(ds.attrs)


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------


def compare_dataset(nc_path: Path, key: str) -> DatasetResult:
    """Compare rsi-tpw epsilon estimates against one ATOMIX benchmark.

    Reads L3 cleaned shear spectra and L4 benchmark epsilon, then
    estimates epsilon from the L3 spectra using rsi-tpw's algorithm.
    """
    info = ATOMIX_DATASETS[key]
    print(f"\n{'─' * 60}")
    print(f"Processing: {info['name']} ({info['instrument']})")
    print(f"  File: {nc_path.name}")

    # Read global attributes for processing parameters
    attrs = read_atomix_attrs(nc_path)
    print(f"  Title: {attrs.get('title', 'N/A')}")

    f_AA = attrs.get("f_AA", attrs.get("HP_cut", 98.0))
    fft_length = attrs.get("fft_length", None)
    goodman = attrs.get("goodman", 1)
    fit_order = 3  # default for rsi-tpw

    print(f"  f_AA={f_AA}, fft_length={fft_length}, goodman={goodman}")

    # Read L3 spectra and L4 dissipation
    try:
        l3 = read_atomix_l3(nc_path)
    except Exception as e:
        print(f"  ERROR reading L3_spectra: {e}")
        return DatasetResult(name=info["name"], key=key, nc_path=nc_path, n_spectra=0, n_probes=0)

    try:
        l4 = read_atomix_l4(nc_path)
    except Exception as e:
        print(f"  ERROR reading L4_dissipation: {e}")
        return DatasetResult(name=info["name"], key=key, nc_path=nc_path, n_spectra=0, n_probes=0)

    # Extract L3 variables — normalize to (n_time, n_wavenum, n_probes) layout.
    # ATOMIX files use varying dimension orders; use named dims to orient.
    sh_var = l3["SH_SPEC_CLEAN"]
    dims = sh_var.dims  # e.g. ('N_SHEAR_SENSORS', 'N_WAVENUMBER', 'TIME_SPECTRA')

    # Identify which axis is which
    time_ax = next(i for i, d in enumerate(dims) if "TIME" in d)
    wn_ax = next(i for i, d in enumerate(dims) if "WAVENUMBER" in d)
    probe_ax = next(i for i, d in enumerate(dims) if "SHEAR" in d)

    # Transpose to (time, wavenumber, probes)
    sh_spec_clean = np.moveaxis(sh_var.values, [time_ax, wn_ax, probe_ax], [0, 1, 2])

    kcyc_var = l3["KCYC"]
    kcyc_raw = kcyc_var.values
    kcyc_dims = kcyc_var.dims
    # Transpose KCYC to (time, wavenumber) if needed
    if len(kcyc_dims) == 2:
        kc_time_ax = next(i for i, d in enumerate(kcyc_dims) if "TIME" in d)
        kc_wn_ax = next(i for i, d in enumerate(kcyc_dims) if "WAVENUMBER" in d)
        kcyc = np.moveaxis(kcyc_raw, [kc_time_ax, kc_wn_ax], [0, 1])
    else:
        kcyc = kcyc_raw

    speed_l3 = l3["PSPD_REL"].values  # (TIME_SPECTRA,)

    n_spectra = sh_spec_clean.shape[0]
    n_wavenum = sh_spec_clean.shape[1]
    n_probes = sh_spec_clean.shape[2]

    # Extract L4 variables — also normalize to (n_time, n_probes)
    epsi_var = l4["EPSI"]
    epsi_dims = epsi_var.dims
    epsi_raw = epsi_var.values
    if len(epsi_dims) == 2:
        ep_time_ax = next(i for i, d in enumerate(epsi_dims) if "TIME" in d)
        ep_probe_ax = next(i for i, d in enumerate(epsi_dims) if "SHEAR" in d)
        epsi = np.moveaxis(epsi_raw, [ep_time_ax, ep_probe_ax], [0, 1])
    else:
        epsi = epsi_raw[:, np.newaxis]

    def _get_l4_2d(name: str, like: np.ndarray = epsi) -> np.ndarray:
        """Read an L4 variable and orient to (time, probes)."""
        if name not in l4:
            return np.full_like(like, np.nan)
        v = l4[name]
        arr = v.values
        if arr.ndim == 1:
            return arr[:, np.newaxis] if arr.shape[0] == n_spectra else arr[np.newaxis, :]
        d = v.dims
        t_ax = next((i for i, dd in enumerate(d) if "TIME" in dd), 0)
        p_ax = next((i for i, dd in enumerate(d) if "SHEAR" in dd), 1)
        return np.moveaxis(arr, [t_ax, p_ax], [0, 1])

    kmin_l4 = _get_l4_2d("KMIN")
    kmax_l4 = _get_l4_2d("KMAX")
    method_l4 = _get_l4_2d("METHOD")
    fom_l4 = _get_l4_2d("FOM")
    kvisc_l4 = l4["KVISC"].values if "KVISC" in l4 else None
    pres_l4 = l4["PRES"].values if "PRES" in l4 else np.full(n_spectra, np.nan)

    print(f"  N_spectra={n_spectra}, N_probes={n_probes}, N_wavenumber={n_wavenum}")
    print(f"  Wavenumber range: {np.nanmin(kcyc):.1f} - {np.nanmax(kcyc):.1f} cpm")
    print(f"  Speed range: {speed_l3.min():.3f} - {speed_l3.max():.3f} m/s")
    print(f"  ATOMIX epsilon range: {np.nanmin(epsi):.2e} - {np.nanmax(epsi):.2e} W/kg")

    result = DatasetResult(
        name=info["name"],
        key=key,
        nc_path=nc_path,
        n_spectra=n_spectra,
        n_probes=n_probes,
    )

    # Process each spectrum through rsi-tpw epsilon estimation
    n_ok = 0
    n_skip = 0
    for i in range(n_spectra):
        # Wavenumber vector (may vary per spectrum or be constant)
        K = kcyc[i, :] if kcyc.ndim == 2 else kcyc

        # Skip if wavenumber is invalid
        if np.all(np.isnan(K)) or len(K) < 3:
            n_skip += 1
            continue

        # Get kinematic viscosity
        if kvisc_l4 is not None and np.isfinite(kvisc_l4[i]):
            nu = float(kvisc_l4[i])
        else:
            # Estimate from temperature if available, else use default
            nu = visc35(10.0)  # ~1.35e-6 m²/s at 10°C

        # Anti-aliasing wavenumber: f_AA / speed
        W = float(speed_l3[i])
        if W <= 0 or not np.isfinite(W):
            n_skip += 1
            continue
        K_AA = f_AA / W

        for p in range(n_probes):
            spec = sh_spec_clean[i, :, p]

            # Skip NaN spectra
            if np.all(np.isnan(spec)) or np.all(spec <= 0):
                n_skip += 1
                continue

            # Replace NaNs with zeros for integration
            spec_clean = np.where(np.isfinite(spec) & (spec > 0), spec, 0.0)
            K_clean = np.where(np.isfinite(K) & (K > 0), K, 1e-10)

            atomix_eps = float(epsi[i, p]) if p < epsi.shape[1] else np.nan

            # Skip if ATOMIX has no estimate
            if not np.isfinite(atomix_eps) or atomix_eps <= 0:
                n_skip += 1
                continue

            # Estimate epsilon using rsi-tpw
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    (
                        eps_rsi,
                        kmax_rsi,
                        mad_rsi,
                        method_rsi,
                        fom_rsi,
                        _var_res,
                        _nas_spec,
                        _kmr,
                        _fm,
                    ) = _estimate_epsilon(
                        K_clean,
                        spec_clean,
                        nu,
                        K_AA,
                        fit_order,
                        e_isr_threshold=E_ISR_THRESHOLD,
                    )
            except Exception as exc:
                print(f"    WARNING: spectrum [{i},{p}] failed: {exc}")
                n_skip += 1
                continue

            result.spectra.append(
                SpectrumResult(
                    idx=i,
                    probe=p,
                    epsilon_atomix=atomix_eps,
                    epsilon_rsi=eps_rsi,
                    kmin_atomix=float(kmin_l4[i, p]) if p < kmin_l4.shape[1] else np.nan,
                    kmax_atomix=float(kmax_l4[i, p]) if p < kmax_l4.shape[1] else np.nan,
                    kmax_rsi=kmax_rsi,
                    method_atomix=int(method_l4[i, p]) if p < method_l4.shape[1] else -1,
                    method_rsi=method_rsi,
                    fom_atomix=float(fom_l4[i, p]) if p < fom_l4.shape[1] else np.nan,
                    fom_rsi=fom_rsi,
                    mad_rsi=mad_rsi,
                    speed=W,
                    pressure=float(pres_l4[i]) if np.isfinite(pres_l4[i]) else np.nan,
                    nu=nu,
                )
            )
            n_ok += 1

    l3.close()
    l4.close()

    print(f"  Processed: {n_ok} spectra, skipped: {n_skip}")
    result.compute_summary()

    if result.spectra:
        print(f"  log10 bias (rsi - atomix): {result.log10_bias:+.3f}")
        print(f"  log10 RMSD:                {result.log10_rmsd:.3f}")
        print(f"  Correlation (log10):       {result.correlation:.3f}")
        print(f"  Within 0.5 decade:         {result.frac_within_half_decade:.1%}")
        print(f"  Within 1.0 decade:         {result.frac_within_1_decade:.1%}")

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_scatter(results: list[DatasetResult], output_dir: Path) -> Path:
    """Scatter plot of rsi-tpw vs ATOMIX epsilon for all datasets."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), squeeze=False)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, res, color in zip(axes[0], results, colors):
        if not res.spectra:
            ax.set_title(f"{res.name}\n(no data)")
            continue

        eps_a = np.array([s.epsilon_atomix for s in res.spectra])
        eps_r = np.array([s.epsilon_rsi for s in res.spectra])

        valid = (eps_a > 0) & (eps_r > 0)
        eps_a, eps_r = eps_a[valid], eps_r[valid]

        lo = min(eps_a.min(), eps_r.min()) * 0.3
        hi = max(eps_a.max(), eps_r.max()) * 3

        ax.loglog(eps_a, eps_r, ".", ms=3, alpha=0.5, color=color)
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, label="1:1")
        ax.plot([lo, hi], [lo * 3, hi * 3], "k--", lw=0.5, alpha=0.4)
        ax.plot([lo, hi], [lo / 3, hi / 3], "k--", lw=0.5, alpha=0.4, label="±0.5 dec")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\epsilon_\mathrm{ATOMIX}$ [W kg$^{-1}$]")
        ax.set_ylabel(r"$\epsilon_\mathrm{rsi\text{-}tpw}$ [W kg$^{-1}$]")
        ax.set_title(
            f"{res.name}\n"
            f"bias={res.log10_bias:+.2f}, RMSD={res.log10_rmsd:.2f}, "
            f"r={res.correlation:.2f}"
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = output_dir / "atomix_epsilon_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved scatter plot: {out}")
    return out


def plot_profiles(results: list[DatasetResult], output_dir: Path) -> Path:
    """Epsilon vs pressure profiles for each dataset."""
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 8), squeeze=False)

    for ax, res in zip(axes[0], results):
        if not res.spectra:
            ax.set_title(f"{res.name}\n(no data)")
            continue

        pressures = np.array([s.pressure for s in res.spectra])
        eps_a = np.array([s.epsilon_atomix for s in res.spectra])
        eps_r = np.array([s.epsilon_rsi for s in res.spectra])

        valid = np.isfinite(pressures) & (eps_a > 0) & (eps_r > 0)
        if valid.sum() == 0:
            ax.set_title(f"{res.name}\n(no pressure data)")
            continue

        ax.semilogx(eps_a[valid], pressures[valid], ".", ms=3, alpha=0.5, label="ATOMIX")
        ax.semilogx(eps_r[valid], pressures[valid], ".", ms=3, alpha=0.5, label="rsi-tpw")
        ax.invert_yaxis()
        ax.set_xlabel(r"$\epsilon$ [W kg$^{-1}$]")
        ax.set_ylabel("Pressure [dbar]")
        ax.set_title(res.name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = output_dir / "atomix_epsilon_profiles.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved profile plot: {out}")
    return out


def plot_kmax_comparison(results: list[DatasetResult], output_dir: Path) -> Path:
    """Compare integration limits (K_max) between ATOMIX and rsi-tpw."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), squeeze=False)

    for ax, res in zip(axes[0], results):
        if not res.spectra:
            ax.set_title(f"{res.name}\n(no data)")
            continue

        kmax_a = np.array([s.kmax_atomix for s in res.spectra])
        kmax_r = np.array([s.kmax_rsi for s in res.spectra])

        valid = np.isfinite(kmax_a) & np.isfinite(kmax_r) & (kmax_a > 0) & (kmax_r > 0)
        if valid.sum() == 0:
            ax.set_title(f"{res.name}\n(no K_max data)")
            continue

        lo = min(kmax_a[valid].min(), kmax_r[valid].min()) * 0.5
        hi = max(kmax_a[valid].max(), kmax_r[valid].max()) * 2

        ax.loglog(kmax_a[valid], kmax_r[valid], ".", ms=3, alpha=0.5)
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$K_\mathrm{max,ATOMIX}$ [cpm]")
        ax.set_ylabel(r"$K_\mathrm{max,rsi\text{-}tpw}$ [cpm]")
        ax.set_title(res.name)
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out = output_dir / "atomix_kmax_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved K_max plot: {out}")
    return out


def plot_fom_comparison(results: list[DatasetResult], output_dir: Path) -> Path:
    """Compare figure-of-merit between ATOMIX and rsi-tpw."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), squeeze=False)

    for ax, res in zip(axes[0], results):
        if not res.spectra:
            ax.set_title(f"{res.name}\n(no data)")
            continue

        fom_a = np.array([s.fom_atomix for s in res.spectra])
        fom_r = np.array([s.fom_rsi for s in res.spectra])

        valid = np.isfinite(fom_a) & np.isfinite(fom_r) & (fom_a > 0) & (fom_r > 0)
        if valid.sum() == 0:
            ax.set_title(f"{res.name}\n(no FOM data)")
            continue

        ax.plot(fom_a[valid], fom_r[valid], ".", ms=3, alpha=0.5)
        lo, hi = 0.5, 2.0
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        ax.axhline(1.15, color="r", ls="--", lw=0.5, alpha=0.5, label="FOM=1.15")
        ax.axvline(1.15, color="r", ls="--", lw=0.5, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("FOM (ATOMIX)")
        ax.set_ylabel("FOM (rsi-tpw)")
        ax.set_title(res.name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "atomix_fom_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved FOM plot: {out}")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

PASS_THRESHOLDS = {
    "log10_rmsd": 0.5,  # Within 0.5 decade RMSD
    "correlation": 0.8,  # log10 correlation > 0.8
    "frac_within_1_decade": 0.9,  # 90% within 1 decade
}


def write_report(
    results: list[DatasetResult],
    output_dir: Path,
    plots: list[Path],
) -> Path:
    """Write a Markdown comparison report."""
    out = output_dir / "ATOMIX_COMPARISON_REPORT.md"

    with open(out, "w") as f:
        f.write("# ATOMIX Benchmark Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## References\n\n")
        f.write(
            "- Fer, I., Dengler, M., Holtermann, P., Le Boyer, A. & Lueck, R. "
            "ATOMIX benchmark datasets for dissipation rate measurements using "
            "shear probes. *Scientific Data* **11**, 518 (2024). "
            "[doi:10.1038/s41597-024-03323-y]"
            "(https://doi.org/10.1038/s41597-024-03323-y)\n"
        )
        f.write(
            "- Lueck, R. et al. Best practices recommendations for estimating "
            "dissipation rates from shear probes. *Front. Mar. Sci.* **11**, "
            "1334327 (2024). [doi:10.3389/fmars.2024.1334327]"
            "(https://doi.org/10.3389/fmars.2024.1334327)\n"
        )
        f.write(
            "- ATOMIX Shear Probes code repository: "
            "[github.com/SCOR-ATOMIX/ShearProbes_BenchmarkDescriptor]"
            "(https://github.com/SCOR-ATOMIX/ShearProbes_BenchmarkDescriptor)\n"
        )
        f.write("\n")

        f.write("## Benchmark Datasets\n\n")
        f.write("| Dataset | Instrument | DOI | Status |\n")
        f.write("|---------|-----------|-----|--------|\n")
        result_map = {r.key: r for r in results}
        for key, info in ATOMIX_DATASETS.items():
            r = result_map.get(key)
            if r and r.spectra:
                status = f"{len(r.spectra)} spectra compared"
            elif r:
                status = "file found, no spectra extracted"
            else:
                status = "not downloaded"
            f.write(
                f"| {info['name']} | {info['instrument']} | "
                f"[{info['doi']}]({BODC_BASE}/{info['doi']}) | "
                f"{status} |\n"
            )

        f.write("\n## Method\n\n")
        f.write(
            "For each ATOMIX benchmark dataset, the L3 cleaned shear spectra "
            "(variable `SH_SPEC_CLEAN`) and corresponding wavenumber vectors "
            "(`KCYC`) are read from the benchmark NetCDF files. Each spectrum "
            "is then processed through the `rsi-tpw` epsilon estimation "
            "algorithm (`_estimate_epsilon` from `rsi.dissipation`), "
            "which implements the Lueck variance method for low dissipation "
            "rates and the inertial subrange method for high dissipation "
            "rates. The resulting epsilon estimates are compared against the "
            "benchmark L4 values (`EPSI`).\n\n"
            "This comparison tests only the spectral fitting / integration "
            "step (L3 -> L4), not the full pipeline from raw data.\n\n"
        )

        f.write("## Pass/Fail Criteria\n\n")
        f.write("| Metric | Threshold | Description |\n")
        f.write("|--------|-----------|-------------|\n")
        f.write(
            f"| log10 RMSD | < {PASS_THRESHOLDS['log10_rmsd']} | "
            "Root-mean-square deviation in log10(epsilon) |\n"
        )
        f.write(
            f"| Correlation | > {PASS_THRESHOLDS['correlation']} | Pearson r of log10(epsilon) |\n"
        )
        f.write(
            f"| Within 1 decade | > {PASS_THRESHOLDS['frac_within_1_decade']:.0%} | "
            "Fraction of estimates within 1 order of magnitude |\n"
        )

        f.write("\n## Results\n\n")

        any_results = False
        overall_pass = True

        for r in results:
            if not r.spectra:
                continue
            any_results = True

            f.write(f"### {r.name} ({r.key})\n\n")
            f.write(f"- **File:** `{r.nc_path.name}`\n")
            f.write(f"- **Spectra compared:** {len(r.spectra)}\n")
            f.write(f"- **Probes:** {r.n_probes}\n\n")

            rmsd_pass = r.log10_rmsd < PASS_THRESHOLDS["log10_rmsd"]
            corr_pass = r.correlation > PASS_THRESHOLDS["correlation"]
            frac_pass = r.frac_within_1_decade > PASS_THRESHOLDS["frac_within_1_decade"]

            ds_pass = rmsd_pass and corr_pass and frac_pass
            if not ds_pass:
                overall_pass = False

            f.write("| Metric | Value | Threshold | Pass |\n")
            f.write("|--------|-------|-----------|------|\n")
            f.write(f"| log10 bias | {r.log10_bias:+.3f} | — | — |\n")
            f.write(
                f"| log10 RMSD | {r.log10_rmsd:.3f} | "
                f"< {PASS_THRESHOLDS['log10_rmsd']} | "
                f"{'PASS' if rmsd_pass else 'FAIL'} |\n"
            )
            f.write(
                f"| Correlation | {r.correlation:.3f} | "
                f"> {PASS_THRESHOLDS['correlation']} | "
                f"{'PASS' if corr_pass else 'FAIL'} |\n"
            )
            f.write(f"| Within 0.5 decade | {r.frac_within_half_decade:.1%} | — | — |\n")
            f.write(
                f"| Within 1 decade | {r.frac_within_1_decade:.1%} | "
                f"> {PASS_THRESHOLDS['frac_within_1_decade']:.0%} | "
                f"{'PASS' if frac_pass else 'FAIL'} |\n"
            )
            f.write(f"\n**Dataset result: {'PASS' if ds_pass else 'FAIL'}**\n\n")

        if any_results:
            f.write("## Overall Result\n\n")
            f.write(f"**{'PASS' if overall_pass else 'FAIL'}**\n\n")
        else:
            f.write("*No benchmark files found. Download datasets and re-run.*\n\n")

        if plots:
            f.write("## Plots\n\n")
            for p in plots:
                f.write(f"![{p.stem}]({p.name})\n\n")

    print(f"\nReport written: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ATOMIX benchmark datasets with rsi-tpw epsilon",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmarks/atomix"),
        help="Directory containing ATOMIX benchmark .nc files (default: benchmarks/atomix/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots and report (default: same as data-dir)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Open browser tabs for downloading benchmark files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(ATOMIX_DATASETS.keys()),
        default=None,
        help="Process only these datasets (default: all found)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir).resolve()

    print("=" * 60)
    print("ATOMIX Benchmark Comparison — rsi-tpw")
    print("=" * 60)
    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")

    # Ensure directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download mode
    if args.download:
        keys = args.datasets or list(ATOMIX_DATASETS.keys())
        for key in keys:
            download_benchmark(data_dir, key)
        print(
            f"\nDownload the NetCDF files from the opened pages and save to:\n"
            f"  {data_dir}/\n"
            f"Then re-run without --download."
        )
        return 0

    # Find benchmark files
    found = find_benchmark_files(data_dir)
    if args.datasets:
        found = {k: v for k, v in found.items() if k in args.datasets}

    if not found:
        print(f"\nNo ATOMIX benchmark files found in {data_dir}/")
        print_download_instructions(data_dir, list(ATOMIX_DATASETS.keys()))
        return 1

    missing = [k for k in ATOMIX_DATASETS if k not in found]
    if missing:
        print(f"\nFound {len(found)}/{len(ATOMIX_DATASETS)} datasets:")
        for k, p in found.items():
            print(f"  {k}: {p.name}")
        print(f"\nMissing: {', '.join(missing)}")

    # Process each dataset
    results = []
    for key, nc_path in found.items():
        try:
            result = compare_dataset(nc_path, key)
            results.append(result)
        except Exception as exc:
            print(f"\n  ERROR processing {key}: {exc}")
            import traceback

            traceback.print_exc()

    # Generate plots
    plots = []
    datasets_with_data = [r for r in results if r.spectra]
    if datasets_with_data and not args.no_plots:
        print(f"\n{'─' * 60}")
        print("Generating plots...")
        plots.append(plot_scatter(datasets_with_data, output_dir))
        plots.append(plot_profiles(datasets_with_data, output_dir))
        plots.append(plot_kmax_comparison(datasets_with_data, output_dir))
        plots.append(plot_fom_comparison(datasets_with_data, output_dir))

    # Write report
    print(f"\n{'─' * 60}")
    write_report(results, output_dir, plots)

    # Print final summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        if r.spectra:
            rmsd_ok = r.log10_rmsd < PASS_THRESHOLDS["log10_rmsd"]
            corr_ok = r.correlation > PASS_THRESHOLDS["correlation"]
            frac_ok = r.frac_within_1_decade > PASS_THRESHOLDS["frac_within_1_decade"]
            status = "PASS" if (rmsd_ok and corr_ok and frac_ok) else "FAIL"
            print(f"  {r.name:30s}  RMSD={r.log10_rmsd:.3f}  r={r.correlation:.3f}  [{status}]")
        else:
            print(f"  {r.name:30s}  (no spectra)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
