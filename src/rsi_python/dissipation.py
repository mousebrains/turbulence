# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Core epsilon (TKE dissipation rate) calculation.

Port of get_diss_odas.m from the ODAS MATLAB library.

References
----------
Lueck, R.G., 2022a: The statistics of oceanic turbulence measurements.
    Part 1: Shear variance and dissipation rates. J. Atmos. Oceanic
    Technol., 39, 1259-1276.
Lueck, R.G., 2022b: The statistics of oceanic turbulence measurements.
    Part 2: Shear spectra and a new spectral model. J. Atmos. Oceanic
    Technol., 39, 1273-1282.
Lueck, R.G., and 27 coauthors, 2024: Best practices recommendations for
    estimating dissipation rates from shear probes. Front. Mar. Sci.,
    11, 1334327.
"""

import re
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

if TYPE_CHECKING:
    from rsi_python.p_file import PFile

from rsi_python.despike import despike
from rsi_python.goodman import clean_shear_spec_batch
from rsi_python.nasmyth import LUECK_A, X_95, nasmyth_grid
from rsi_python.ocean import visc, visc35

# ---------------------------------------------------------------------------
# Named constants — citations inline
# ---------------------------------------------------------------------------

# Minimum epsilon floor to avoid zero/negative values [W/kg]
EPSILON_FLOOR = 1e-15

# Minimum profiling speed to avoid wavenumber singularity [m/s]
SPEED_MIN = 0.01

# Epsilon threshold for switching from variance to ISR method [W/kg]
E_ISR_THRESHOLD = 1.5e-5

# Wavenumber limits for spectral integration [cpm]
K_LIMIT_MIN = 7  # minimum upper integration limit
K_LIMIT_MAX = 150  # maximum upper integration limit
K_INITIAL_CUTOFF = 10  # initial integration cutoff for e_10 estimate

# Isotropy factor: epsilon = ISOTROPY_FACTOR * nu * integral(phi_shear)
# Tennekes & Lumley (1972)
ISOTROPY_FACTOR = 7.5

# Nuttall (1971) DOF correction for cosine-windowed overlapped FFTs
DOF_NUTTALL = 1.9

# ISR wavenumber limit factor (2 * 0.01, matches ODAS)
X_ISR = 0.01

# Macoun & Lueck (1998) wavenumber correction: 1 + (K/MACOUN_LUECK_DENOM)^2
# Applied for K <= MACOUN_LUECK_K
MACOUN_LUECK_K = 150  # [cpm]
MACOUN_LUECK_DENOM = 48  # [cpm]

# Variance correction convergence threshold (< 2% change, matches ODAS)
VARIANCE_CONVERGENCE = 1.02

# Variance correction model coefficients (Lueck, ODAS lines 623-634)
VARIANCE_TANH_COEFF = 48
VARIANCE_EXP_COEFF = 2.9
VARIANCE_EXP_DECAY = 22.3

# Low-wavenumber correction ratio threshold (ODAS lines 636-638)
LOW_K_CORRECTION_THRESHOLD = 1.1
LOW_K_CORRECTION_FACTOR = 0.25

# ISR flyer detection threshold [decades]
ISR_FLYER_THRESHOLD = 0.5
# Maximum fraction of points to remove as flyers
ISR_FLYER_FRAC = 0.2

# Lueck (2022a,b) FM statistic coefficients
FM_SIGMA_COEFF = 1.25  # 5/4
FM_SIGMA_EXPONENT = -7 / 9
FM_TM_OFFSET = 0.8
FM_TM_COEFF = 1.56

# Goodman bias correction factor (ODAS TN-061)
GOODMAN_BIAS = 1.02

# Anti-aliasing safety margin (ODAS convention)
F_AA_MARGIN = 0.9

# Effective speed cutout for profile detection [m/s]
SPEED_CUTOUT = 0.05

# ---------------------------------------------------------------------------
# Channel loading
# ---------------------------------------------------------------------------


def load_channels(
    source: "PFile | str | Path",
    shear_pattern: str = r"^sh\d+$",
    accel_pattern: str = r"^A[xyz]$",
    pressure_name: str = "P",
    temperature_name: str = "T1",
) -> dict[str, Any]:
    """Load channel data from any supported source.

    Parameters
    ----------
    source : PFile, str, or Path
        A PFile object, .p file path, full-record .nc path,
        or per-profile .nc path.
    shear_pattern : str
        Regex pattern matching shear channel names.
    accel_pattern : str
        Regex pattern matching accelerometer channel names.
    pressure_name : str
        Name of the pressure channel.
    temperature_name : str
        Name of the temperature channel.

    Returns
    -------
    dict with keys:
        shear : list of (name, ndarray) — shear probe signals
        accel : list of (name, ndarray) — accelerometer signals
        P : ndarray — pressure (slow)
        T : ndarray — temperature (slow)
        t_fast : ndarray — fast time vector
        t_slow : ndarray — slow time vector
        fs_fast : float
        fs_slow : float
        is_profile : bool — whether source is a per-profile file
        metadata : dict
    """
    from rsi_python.p_file import PFile

    if isinstance(source, PFile):
        return _channels_from_pfile(
            source, shear_pattern, accel_pattern, pressure_name, temperature_name
        )

    source = Path(source)
    if source.suffix.lower() == ".p":
        pf = PFile(source)
        return _channels_from_pfile(
            pf, shear_pattern, accel_pattern, pressure_name, temperature_name
        )
    elif source.suffix.lower() == ".nc":
        return _channels_from_nc(
            source, shear_pattern, accel_pattern, pressure_name, temperature_name
        )
    else:
        raise ValueError(f"Unsupported file type: {source.suffix}")


def _channels_from_pfile(
    pf: "PFile", sh_pat: str, ac_pat: str, p_name: str, t_name: str
) -> dict[str, Any]:
    sh_re = re.compile(sh_pat)
    ac_re = re.compile(ac_pat)
    shear = sorted(
        [(n, pf.channels[n]) for n in pf._fast_channels if sh_re.match(n)],
        key=lambda x: x[0],
    )
    accel = sorted(
        [(n, pf.channels[n]) for n in pf._fast_channels if ac_re.match(n)],
        key=lambda x: x[0],
    )
    return {
        "shear": shear,
        "accel": accel,
        "P": pf.channels[p_name],
        "T": pf.channels[t_name],
        "t_fast": pf.t_fast,
        "t_slow": pf.t_slow,
        "fs_fast": pf.fs_fast,
        "fs_slow": pf.fs_slow,
        "is_profile": False,
        "vehicle": pf.config["instrument_info"].get("vehicle", "").lower(),
        "metadata": {
            "source": str(pf.filepath),
            "instrument": pf.config["instrument_info"].get("model", ""),
            "sn": pf.config["instrument_info"].get("sn", ""),
            "start_time": pf.start_time.isoformat(),
        },
    }


def _channels_from_nc(
    nc_path: Path, sh_pat: str, ac_pat: str, p_name: str, t_name: str
) -> dict[str, Any]:
    import netCDF4 as nc

    ds = nc.Dataset(str(nc_path), "r")
    sh_re = re.compile(sh_pat)
    ac_re = re.compile(ac_pat)

    fs_fast = float(ds.fs_fast)
    fs_slow = float(ds.fs_slow)
    is_profile = hasattr(ds, "profile_number")

    t_fast = ds.variables["t_fast"][:].data
    t_slow = ds.variables["t_slow"][:].data
    P = ds.variables[p_name][:].data.astype(np.float64)
    T = ds.variables[t_name][:].data.astype(np.float64)

    shear = []
    accel = []
    for vname in sorted(ds.variables.keys()):
        var = ds.variables[vname]
        if var.dimensions == ("time_fast",):
            data = var[:].data.astype(np.float64)
            if sh_re.match(vname):
                shear.append((vname, data))
            elif ac_re.match(vname):
                accel.append((vname, data))

    metadata = {"source": str(nc_path)}
    for attr in ("instrument_model", "instrument_sn", "source_file", "start_time"):
        if hasattr(ds, attr):
            metadata[attr] = getattr(ds, attr)

    vehicle = ""
    if hasattr(ds, "instrument_model"):
        model = ds.instrument_model.lower()
        if "vmp" in model:
            vehicle = "vmp"
        elif "xmp" in model:
            vehicle = "xmp"

    ds.close()

    return {
        "shear": shear,
        "accel": accel,
        "P": P,
        "T": T,
        "t_fast": t_fast,
        "t_slow": t_slow,
        "fs_fast": fs_fast,
        "fs_slow": fs_slow,
        "is_profile": is_profile,
        "vehicle": vehicle,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Shared helpers (used by both dissipation.py and chi.py)
# ---------------------------------------------------------------------------


def _prepare_profiles(
    data: dict[str, Any],
    speed: float | None,
    direction: str,
    salinity: npt.ArrayLike | None,
    tau: float = 1.5,
    speed_cutout: float = 0.05,
) -> tuple | None:
    """Profile detection, speed computation, and salinity interpolation.

    Matches the ODAS odas_p2mat.m speed pipeline:
      1. W = gradient(P) filtered with Butterworth at 0.68/tau
      2. speed = abs(W)
      3. speed filtered again with Butterworth at 0.68/tau
      4. speed clamped to speed_cutout minimum

    Returns (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast,
    fs_slow, ratio).
    """
    from scipy.signal import butter, filtfilt

    fs_fast = data["fs_fast"]
    fs_slow = data["fs_slow"]
    ratio = round(fs_fast / fs_slow)

    P_slow = data["P"]
    T_slow = data["T"]
    t_fast = data["t_fast"]
    t_slow = data["t_slow"]

    if data["is_profile"]:
        profiles_slow = [(0, len(P_slow) - 1)]
    else:
        from rsi_python.profile import _smooth_fall_rate, get_profiles

        W_slow = _smooth_fall_rate(P_slow, fs_slow, tau=tau)
        profiles_slow = get_profiles(P_slow, W_slow, fs_slow, direction=direction)
        if not profiles_slow:
            return None

    if speed is not None:
        speed_fast = np.full(len(t_fast), abs(speed))
    else:
        from rsi_python.profile import _smooth_fall_rate

        W_slow = _smooth_fall_rate(P_slow, fs_slow, tau=tau)

        # VMP pressure-based: speed = abs(W), matching odas_p2mat.m l.757
        speed_slow = np.abs(W_slow)

        # Interpolate to fast rate before second smoothing pass
        speed_fast = np.interp(t_fast, t_slow, speed_slow)

        # Second Butterworth smoothing pass on speed, odas_p2mat.m l.898-901
        f_c = 0.68 / tau
        b_slow, a_slow = butter(1, f_c / (fs_slow / 2.0))
        speed_slow = filtfilt(b_slow, a_slow, speed_slow)
        b_fast, a_fast = butter(1, f_c / (fs_fast / 2.0))
        speed_fast = filtfilt(b_fast, a_fast, speed_fast)

        # Clamp to minimum, odas_p2mat.m l.905-906
        speed_slow[speed_slow < speed_cutout] = speed_cutout
        speed_fast[speed_fast < speed_cutout] = speed_cutout

    P_fast = np.interp(t_fast, t_slow, P_slow)
    T_fast = np.interp(t_fast, t_slow, T_slow)

    if salinity is not None:
        salinity = np.asarray(salinity, dtype=float)
        if salinity.ndim > 0:
            if len(salinity) == len(t_slow):
                sal_fast = np.interp(t_fast, t_slow, salinity)
            elif len(salinity) == len(t_fast):
                sal_fast = salinity
            else:
                raise ValueError(
                    f"salinity array length {len(salinity)} doesn't match "
                    f"slow ({len(t_slow)}) or fast ({len(t_fast)}) time series"
                )
        else:
            sal_fast = float(salinity)
    else:
        sal_fast = None

    return (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, fs_slow, ratio, t_fast)


def _compute_nu(
    mean_T: float, mean_P: float, salinity: float | np.ndarray | None, sel: slice
) -> float:
    """Compute kinematic viscosity, dispatching to visc35 or visc.

    Parameters
    ----------
    mean_T : float
    mean_P : float
    salinity : float, ndarray, or None
    sel : slice
        Used to index salinity when it's an array.
    """
    if salinity is not None:
        mean_S = float(np.mean(salinity[sel]) if np.ndim(salinity) > 0 else salinity)
        return float(visc(mean_T, mean_S, mean_P))
    return float(visc35(mean_T))


# ---------------------------------------------------------------------------
# Core dissipation calculation
# ---------------------------------------------------------------------------


def get_diss(
    source: "PFile | str | Path",
    fft_length: int = 256,
    diss_length: int | None = None,
    overlap: int | None = None,
    speed: float | None = None,
    direction: str = "down",
    goodman: bool = True,
    f_AA: float = 98.0,
    f_limit: float | None = None,
    fit_order: int = 3,
    despike_thresh: float = 8,
    despike_smooth: float = 0.5,
    salinity: npt.ArrayLike | None = None,
) -> list[xr.Dataset]:
    """Compute epsilon from any source.

    Parameters
    ----------
    source : PFile, str, or Path
        A PFile object, .p file path, full-record .nc,
        or per-profile .nc.
    fft_length : int
        FFT segment length in samples. Default: 256 (0.5 s at 512 Hz).
    diss_length : int or None
        Dissipation window in samples. Default: 2 * fft_length.
    overlap : int or None
        Window overlap in samples. Default: diss_length // 2.
    speed : float or None
        Profiling speed [m/s]. If None, computed from dP/dt.
    direction : str
        'up' or 'down' for speed sign convention.
    goodman : bool
        Apply Goodman coherent noise removal. Default: True.
    f_AA : float
        Anti-aliasing filter cutoff [Hz]. Default: 98.
    f_limit : float or None
        Maximum frequency for integration. Default: None (use f_AA).
    fit_order : int
        Polynomial order for spectral minimum fit. Default: 3.
    despike_thresh : float
        Despike threshold. Default: 8.
    despike_smooth : float
        Despike smoothing frequency [Hz]. Default: 0.5.
    salinity : float or array_like or None
        Practical salinity [PSU]. If provided, uses gsw-based viscosity
        instead of visc35. Scalar or array matching slow time series.

    Returns
    -------
    list of xarray.Dataset
        One Dataset per profile. Each contains:
        epsilon, K_max, speed, nu, T_mean, P_mean, spec_shear,
        spec_nasmyth, K, mad, fom, K_max_ratio.
    """
    if diss_length is None:
        diss_length = 2 * fft_length
    if overlap is None:
        overlap = diss_length // 2
    if f_limit is None:
        f_limit = np.inf

    data = load_channels(source)

    # Get shear and accel arrays
    shear_names = [s[0] for s in data["shear"]]
    shear_arrays = [s[1] for s in data["shear"]]
    accel_arrays = [a[1] for a in data["accel"]]
    n_shear = len(shear_arrays)
    n_accel = len(accel_arrays)

    if n_shear == 0:
        raise ValueError("No shear channels found")
    if goodman and n_accel == 0:
        raise ValueError("No accelerometer channels found (required for Goodman)")

    # Despike shear data
    fs_fast = data["fs_fast"]
    for i in range(n_shear):
        shear_arrays[i], _, _, _ = despike(
            shear_arrays[i],
            fs_fast,
            thresh=despike_thresh,
            smooth=despike_smooth,
        )

    # Look up vehicle-specific tau for speed smoothing
    from rsi_python.profile import _VEHICLE_TAU

    vehicle = data.get("vehicle", "")
    tau = _VEHICLE_TAU.get(vehicle, 1.5)

    # Profile detection, speed, P/T interpolation, salinity
    prepared = _prepare_profiles(data, speed, direction, salinity, tau=tau)
    if prepared is None:
        return []
    (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, _fs_slow, ratio, t_fast) = (
        prepared
    )

    # Convert shear from piezo output to du/dz by dividing by speed^2,
    # matching ODAS odas_p2mat.m line 922.
    for i in range(n_shear):
        shear_arrays[i] = shear_arrays[i] / speed_fast**2

    # Effective AA cutoff
    f_AA_eff = F_AA_MARGIN * f_AA
    if f_limit < f_AA_eff:
        f_AA_eff = f_limit

    results = []
    for s_slow, e_slow in profiles_slow:
        s_fast = s_slow * ratio
        e_fast = min((e_slow + 1) * ratio, len(t_fast))

        # Build shear and accel matrices for this profile
        sh_mat = np.column_stack([s[s_fast:e_fast] for s in shear_arrays])
        ac_mat = np.column_stack([a[s_fast:e_fast] for a in accel_arrays])
        p_prof = P_fast[s_fast:e_fast]
        t_prof = T_fast[s_fast:e_fast]
        spd_prof = speed_fast[s_fast:e_fast]
        time_prof = t_fast[s_fast:e_fast]

        sal_prof = sal_fast[s_fast:e_fast] if isinstance(sal_fast, np.ndarray) else sal_fast

        ds = _compute_profile_diss(
            sh_mat,
            ac_mat,
            p_prof,
            t_prof,
            spd_prof,
            time_prof,
            shear_names,
            fs_fast,
            fft_length,
            diss_length,
            overlap,
            goodman,
            f_AA_eff,
            fit_order,
            salinity=sal_prof,
        )
        ds.attrs.update(data["metadata"])
        ds.attrs["history"] = (
            f"Computed with rsi-python on {datetime.now(UTC).isoformat()}"
        )
        # CF time coordinate attributes
        start_time = data["metadata"].get("start_time", "")
        t_units = f"seconds since {start_time}" if start_time else "seconds"
        ds.coords["t"].attrs.update(
            {
                "standard_name": "time",
                "long_name": "time of dissipation estimate",
                "units": t_units,
                "calendar": "standard",
                "axis": "T",
            }
        )
        results.append(ds)

    return results


def _compute_profile_diss(
    shear,
    accel,
    P,
    T,
    speed,
    t_fast,
    shear_names,
    fs_fast,
    fft_length,
    diss_length,
    overlap,
    do_goodman,
    f_AA,
    fit_order,
    salinity=None,
):
    """Compute dissipation for all windows in a single profile.

    Iterates over dissipation windows, computes Goodman-cleaned shear
    wavenumber spectra, and estimates epsilon via the variance or ISR method.

    Parameters
    ----------
    shear : ndarray, shape (N, n_shear)
        Shear probe data [s⁻¹], already despiked and divided by speed².
    accel : ndarray, shape (N, n_accel)
        Accelerometer data for Goodman cleaning.
    P : ndarray, shape (N,)
        Pressure [dbar].
    T : ndarray, shape (N,)
        Temperature [°C].
    speed : ndarray, shape (N,)
        Profiling speed [m/s].
    t_fast : ndarray, shape (N,)
        Time vector [s].
    shear_names : list of str
        Probe names for coordinate labelling.
    fs_fast : float
        Sampling rate [Hz].
    fft_length : int
        FFT segment length [samples].
    diss_length : int
        Dissipation window length [samples].
    overlap : int
        Window overlap [samples].
    do_goodman : bool
        Whether to apply Goodman coherent noise removal.
    f_AA : float
        Effective anti-aliasing cutoff [Hz].
    fit_order : int
        Polynomial order for spectral minimum detection.
    salinity : float, ndarray, or None
        Practical salinity for viscosity calculation.

    Returns
    -------
    xarray.Dataset
        Contains epsilon, K_max, mad, fom, FM, K_max_ratio, spectra, etc.
    """
    N = shear.shape[0]
    n_shear = shear.shape[1]
    n_freq = fft_length // 2 + 1

    # Number of dissipation estimates
    if overlap >= diss_length:
        overlap = diss_length // 2
    step = diss_length - overlap
    n_est = max(1, 1 + (N - diss_length) // step)

    # Trim n_est if the last window would exceed the data
    while n_est > 0 and (n_est - 1) * step + diss_length > N:
        n_est -= 1

    # Degrees of freedom — Lueck (2022a,b)
    num_ffts = 2 * (diss_length // fft_length) - 1
    n_v = accel.shape[1] if do_goodman else 0
    dof_spec = DOF_NUTTALL * max(num_ffts - n_v, 1)

    # ---- Vectorized: extract all windows at once ----
    win_starts = np.arange(n_est) * step
    indices = win_starts[:, np.newaxis] + np.arange(diss_length)[np.newaxis, :]
    # shape: (n_est, diss_length, n_channels)
    sh_windows = shear[indices]
    ac_windows = accel[indices]

    # ---- Vectorized: batched CSD + Goodman ----
    if do_goodman:
        # clean_UU: (n_est, n_freq, n_shear, n_shear)
        P_sh_clean_all, F = clean_shear_spec_batch(
            ac_windows, sh_windows, fft_length, fs_fast,
        )
    else:
        from rsi_python.spectral import csd_matrix_batch

        P_sh_all, F, _, _ = csd_matrix_batch(
            sh_windows, None, fft_length, fs_fast,
            overlap=fft_length // 2, detrend="linear",
        )
        P_sh_clean_all = np.real(P_sh_all)

    del sh_windows, ac_windows

    # ---- Vectorized: window means ----
    # Reshape for fast window-mean computation
    speed_windows = np.abs(speed[indices])  # (n_est, diss_length)
    speed_out = np.mean(speed_windows, axis=1)
    speed_out = np.maximum(speed_out, SPEED_MIN)

    T_out = np.mean(T[indices], axis=1)
    P_out = np.mean(P[indices], axis=1)
    t_out = np.mean(t_fast[indices], axis=1)

    # Viscosity per window
    if salinity is not None:
        if np.ndim(salinity) > 0:
            sal_windows = salinity[indices]
            sal_means = np.mean(sal_windows, axis=1)
        else:
            sal_means = np.full(n_est, float(salinity))
        nu_out = np.array([float(visc(T_out[i], sal_means[i], P_out[i])) for i in range(n_est)])
    else:
        nu_out = np.array([float(visc35(T_out[i])) for i in range(n_est)])

    # ---- Vectorized: frequency → wavenumber conversion + correction ----
    # K varies per window because speed varies: K[f, w] = F[f] / W[w]
    K_all = F[:, np.newaxis] / speed_out[np.newaxis, :]  # (n_freq, n_est)
    K_AA_all = f_AA / speed_out  # (n_est,)

    # Macoun-Lueck correction: (n_freq, n_est)
    correction = np.ones_like(K_all)
    mask = K_all <= MACOUN_LUECK_K
    correction[mask] = 1 + (K_all[mask] / MACOUN_LUECK_DENOM) ** 2

    # Convert CSD from frequency to wavenumber spectra and apply correction
    # P_sh_clean_all: (n_est, n_freq, n_sh, n_sh)
    # Multiply by W (per window) and correction (per freq, per window)
    wk_scale = speed_out[np.newaxis, :] * correction  # (n_freq, n_est)
    P_sh_clean_all = P_sh_clean_all * wk_scale.T[:, :, np.newaxis, np.newaxis]

    # Store F (same for all windows)
    F_out = np.broadcast_to(F[:, np.newaxis], (n_freq, n_est)).copy()
    K_out = K_all

    # ---- Per-window epsilon estimation (not vectorizable) ----
    epsilon = np.full((n_shear, n_est), np.nan)
    K_max_out = np.full((n_shear, n_est), np.nan)
    mad_out = np.full((n_shear, n_est), np.nan)
    fom_out = np.full((n_shear, n_est), np.nan)
    FM_out = np.full((n_shear, n_est), np.nan)
    K_max_ratio_out = np.full((n_shear, n_est), np.nan)
    method_out = np.zeros((n_shear, n_est), dtype=np.int8)
    spec_shear = np.full((n_shear, n_freq, n_est), np.nan)
    spec_nasmyth = np.full((n_shear, n_freq, n_est), np.nan)

    for idx in range(n_est):
        K = K_all[:, idx]
        K_AA = K_AA_all[idx]
        nu = nu_out[idx]

        for ci in range(n_shear):
            shear_spec = np.real(P_sh_clean_all[idx, :, ci, ci])

            e_4, k_max, mad, meth, nas_spec, fom_val, K_max_ratio_val, FM_val = _estimate_epsilon(
                K,
                shear_spec,
                nu,
                K_AA,
                fit_order,
                E_ISR_THRESHOLD,
                num_ffts=num_ffts,
                n_v=n_v,
            )
            epsilon[ci, idx] = e_4
            K_max_out[ci, idx] = k_max
            mad_out[ci, idx] = mad
            fom_out[ci, idx] = fom_val
            FM_out[ci, idx] = FM_val
            K_max_ratio_out[ci, idx] = K_max_ratio_val
            method_out[ci, idx] = meth
            spec_shear[ci, :, idx] = shear_spec
            spec_nasmyth[ci, :, idx] = nas_spec

    return _build_diss_dataset(
        epsilon=epsilon,
        K_max_out=K_max_out,
        mad_out=mad_out,
        fom_out=fom_out,
        FM_out=FM_out,
        K_max_ratio_out=K_max_ratio_out,
        method_out=method_out,
        speed_out=speed_out,
        nu_out=nu_out,
        P_out=P_out,
        T_out=T_out,
        t_out=t_out,
        spec_shear=spec_shear,
        spec_nasmyth=spec_nasmyth,
        K_out=K_out,
        F_out=F_out,
        shear_names=shear_names,
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        fs_fast=fs_fast,
        dof_spec=dof_spec,
        do_goodman=do_goodman,
        f_AA=f_AA,
        fit_order=fit_order,
    )


def _build_diss_dataset(
    *,
    epsilon: np.ndarray,
    K_max_out: np.ndarray,
    mad_out: np.ndarray,
    fom_out: np.ndarray,
    FM_out: np.ndarray,
    K_max_ratio_out: np.ndarray,
    method_out: np.ndarray,
    speed_out: np.ndarray,
    nu_out: np.ndarray,
    P_out: np.ndarray,
    T_out: np.ndarray,
    t_out: np.ndarray,
    spec_shear: np.ndarray,
    spec_nasmyth: np.ndarray,
    K_out: np.ndarray,
    F_out: np.ndarray,
    shear_names: list[str],
    fft_length: int,
    diss_length: int,
    overlap: int,
    fs_fast: float,
    dof_spec: float,
    do_goodman: bool,
    f_AA: float,
    fit_order: int,
) -> xr.Dataset:
    """Build an xarray Dataset from epsilon estimation output arrays."""
    ds = xr.Dataset(
        {
            "epsilon": (
                ["probe", "time"],
                epsilon,
                {
                    "units": "W kg-1",
                    "long_name": "TKE dissipation rate",
                },
            ),
            "K_max": (
                ["probe", "time"],
                K_max_out,
                {
                    "units": "cpm",
                    "long_name": "upper wavenumber integration limit",
                },
            ),
            "mad": (
                ["probe", "time"],
                mad_out,
                {
                    "units": "1",
                    "long_name": "mean absolute deviation of spectral fit in log10 space",
                },
            ),
            "fom": (
                ["probe", "time"],
                fom_out,
                {
                    "units": "1",
                    "long_name": "figure of merit (observed/Nasmyth variance ratio)",
                },
            ),
            "FM": (
                ["probe", "time"],
                FM_out,
                {
                    "units": "1",
                    "long_name": "Lueck figure of merit (MAD * sqrt(dof))",
                    "comment": "FM < 1 for 97.5% of good spectra (Lueck, 2022a,b)",
                },
            ),
            "K_max_ratio": (
                ["probe", "time"],
                K_max_ratio_out,
                {
                    "units": "1",
                    "long_name": "K_max / K_95 spectral resolution ratio",
                },
            ),
            "method": (
                ["probe", "time"],
                method_out,
                {
                    "long_name": "spectral fitting method",
                    "flag_values": np.array([0, 1], dtype=np.int8),
                    "flag_meanings": "variance inertial_subrange",
                },
            ),
            "speed": (
                ["time"],
                speed_out,
                {
                    "units": "m s-1",
                    "long_name": "profiling speed",
                },
            ),
            "nu": (
                ["time"],
                nu_out,
                {
                    "units": "m2 s-1",
                    "long_name": "kinematic viscosity of sea water",
                },
            ),
            "P_mean": (
                ["time"],
                P_out,
                {
                    "units": "dbar",
                    "long_name": "mean sea water pressure",
                    "standard_name": "sea_water_pressure",
                    "positive": "down",
                },
            ),
            "T_mean": (
                ["time"],
                T_out,
                {
                    "units": "degree_Celsius",
                    "long_name": "mean sea water temperature",
                    "standard_name": "sea_water_temperature",
                },
            ),
            "spec_shear": (
                ["probe", "freq", "time"],
                spec_shear,
                {
                    "units": "s-2 cpm-1",
                    "long_name": "shear wavenumber spectrum (observed, cleaned)",
                },
            ),
            "spec_nasmyth": (
                ["probe", "freq", "time"],
                spec_nasmyth,
                {
                    "units": "s-2 cpm-1",
                    "long_name": "Nasmyth theoretical shear spectrum",
                },
            ),
            "K": (
                ["freq", "time"],
                K_out,
                {
                    "units": "cpm",
                    "long_name": "wavenumber (cycles per metre)",
                },
            ),
            "F": (
                ["freq", "time"],
                F_out,
                {
                    "units": "Hz",
                    "long_name": "frequency",
                },
            ),
        },
        coords={
            "probe": shear_names,
            "t": (["time"], t_out),
        },
        attrs={
            "Conventions": "CF-1.13",
            "fft_length": fft_length,
            "diss_length": diss_length,
            "overlap": overlap,
            "fs_fast": fs_fast,
            "dof_spec": dof_spec,
            "goodman": int(do_goodman),
            "f_AA": f_AA,
            "fit_order": fit_order,
        },
    )
    ds.coords["probe"].attrs["long_name"] = "shear probe name"
    return ds


def _estimate_epsilon(
    K: np.ndarray,
    shear_spectrum: np.ndarray,
    nu: float,
    K_AA: float,
    fit_order: int,
    e_isr_threshold: float,
    num_ffts: int = 3,
    n_v: int = 0,
) -> tuple[float, float, float, int, np.ndarray, float, float, float]:
    """Estimate epsilon from a single shear wavenumber spectrum.

    Implements the ODAS variance method (with polynomial spectral-minimum
    detection and iterative variance correction) for low dissipation, and
    the inertial-subrange (ISR) method for high dissipation.

    Parameters
    ----------
    K : ndarray
        Wavenumber vector [cpm].
    shear_spectrum : ndarray
        Observed shear spectrum [s⁻² cpm⁻¹].
    nu : float
        Kinematic viscosity [m²/s].
    K_AA : float
        Anti-aliasing wavenumber limit [cpm].
    fit_order : int
        Polynomial order for spectral minimum detection.
    e_isr_threshold : float
        Epsilon threshold for switching to ISR method [W/kg].
    num_ffts : int
        Number of FFT segments (for Lueck FM statistic).
    n_v : int
        Number of vibration signals removed (for Lueck FM statistic).

    Returns
    -------
    epsilon : float
        Dissipation rate [W/kg].
    K_max : float
        Upper integration limit [cpm].
    mad : float
        Mean absolute deviation of spectral fit (log10).
    method : int
        0 = variance, 1 = inertial subrange.
    nasmyth_spectrum : ndarray
        Nasmyth spectrum at final epsilon.
    fom : float
        Figure of merit (observed/Nasmyth variance ratio).
    K_max_ratio : float
        K_max / K_95 spectral resolution ratio.
    FM : float
        Lueck (2022a,b) figure of merit.
    """
    n_freq = len(K)
    method = 0

    # Initial estimate: integrate to K_INITIAL_CUTOFF cpm, use e/e_10 model
    K_range = np.where(K <= K_INITIAL_CUTOFF)[0]
    if len(K_range) < 3:
        K_range = np.arange(min(3, n_freq))

    e_10 = ISOTROPY_FACTOR * nu * np.trapezoid(shear_spectrum[K_range], K[K_range])
    if e_10 <= 0:
        warnings.warn(
            f"Initial epsilon estimate e_10={e_10:.2e} <= 0; clamped to {EPSILON_FLOOR:.0e}",
            stacklevel=2,
        )
        e_10 = EPSILON_FLOOR
    e_1 = e_10 * np.sqrt(1 + LUECK_A * e_10)

    if e_1 < e_isr_threshold:
        # Variance method
        # Check if enough points for ISR refinement
        isr_limit = X_ISR * (e_1 / nu**3) ** 0.25
        if len(np.where(isr_limit >= K)[0]) >= 20:
            e_2, _ = _inertial_subrange(K, shear_spectrum, e_1, nu, min(K_LIMIT_MAX, K_AA))
        else:
            e_2 = e_1

        K_95 = X_95 * (e_2 / nu**3) ** 0.25
        valid_K_limit = min(K_AA, K_95)
        valid_K_limit = np.clip(valid_K_limit, K_LIMIT_MIN, K_LIMIT_MAX)
        valid_idx = np.where(valid_K_limit >= K)[0]
        index_limit = len(valid_idx)

        if index_limit <= 1:
            index_limit = min(3, n_freq)

        # Polynomial fit to find spectral minimum
        y = np.log10(shear_spectrum[1:index_limit] + 1e-30)
        x = np.log10(K[1:index_limit] + 1e-30)

        fit_order_eff = np.clip(fit_order, 3, 8)
        K_limit_log = np.log10(K_95)

        if index_limit > fit_order_eff + 2:
            # Remove NaNs
            valid = np.isfinite(y) & np.isfinite(x)
            if np.sum(valid) > fit_order_eff + 2:
                p = np.polyfit(x[valid], y[valid], fit_order_eff)
                pd1 = np.polyder(p)
                roots = np.roots(pd1)
                # Keep only real roots
                roots = roots[np.isreal(roots)].real
                # Keep minima (second derivative > 0)
                pd2 = np.polyder(pd1)
                roots = roots[np.polyval(pd2, roots) > 0]
                # Keep roots above K_INITIAL_CUTOFF cpm
                roots = roots[roots >= np.log10(K_INITIAL_CUTOFF)]
                K_limit_log = roots[0] if len(roots) > 0 else np.log10(K_95)
            else:
                K_limit_log = np.log10(K_95)
        else:
            K_limit_log = np.log10(K_95)

        # Final integration limit
        K_limit_log = min(K_limit_log, np.log10(K_95), np.log10(K_AA))
        K_limit_log = np.clip(K_limit_log, np.log10(K_LIMIT_MIN), np.log10(K_LIMIT_MAX))

        Range = np.where(10**K_limit_log >= K)[0]
        if len(Range) > 0 and K[Range[-1]] < K_LIMIT_MIN:
            # Extend to include at least 7 cpm
            Range = np.append(Range, Range[-1] + 1)
        if len(Range) < 3:
            Range = np.arange(min(3, n_freq))

        e_3 = ISOTROPY_FACTOR * nu * np.trapezoid(shear_spectrum[Range], K[Range])
        e_3 = max(e_3, EPSILON_FLOOR)

        # Iterative variance correction
        e_4 = _variance_correction(e_3, K[Range[-1]], nu)

        # Correct for missing variance at bottom end (odas l.635-638)
        if len(K) > 2:
            e_4_vc = e_4  # variance-corrected value for ratio test
            phi_low = nasmyth_grid(e_4, nu, K[1:3])
            e_4 = e_4 + LOW_K_CORRECTION_FACTOR * ISOTROPY_FACTOR * nu * K[1] * phi_low[0]
            if e_4 / e_4_vc > LOW_K_CORRECTION_THRESHOLD:
                e_4 = _variance_correction(e_4, K[Range[-1]], nu)

        k_max = K[Range[-1]]

    else:
        # Inertial subrange method
        method = 1
        K_limit = min(K_AA, K_LIMIT_MAX)
        e_4, k_max = _inertial_subrange(K, shear_spectrum, e_1, nu, K_limit)
        Range = np.where(k_max >= K)[0]
        if len(Range) < 3:
            Range = np.arange(min(3, n_freq))

    # Compute Nasmyth spectrum at final epsilon
    nas_spec = nasmyth_grid(max(e_4, EPSILON_FLOOR), nu, K + 1e-30)

    # Mean absolute deviation (log10, matching ODAS convention)
    if len(Range) > 1:
        spec_ratio = shear_spectrum[Range[1:]] / (nas_spec[Range[1:]] + 1e-30)
        spec_ratio = spec_ratio[spec_ratio > 0]
        mad = float(np.mean(np.abs(np.log10(spec_ratio)))) if len(spec_ratio) > 0 else float("nan")
    else:
        mad = float("nan")

    # Figure of merit: observed/Nasmyth variance ratio in fit range
    if len(Range) > 1:
        obs_var = np.trapezoid(shear_spectrum[Range[1:]], K[Range[1:]])
        nas_var = np.trapezoid(nas_spec[Range[1:]], K[Range[1:]])
        fom = obs_var / nas_var if nas_var > 0 else np.nan
    else:
        fom = np.nan

    # Lueck (2022a,b) figure of merit — FM < 1 for 97.5% of good spectra.
    #   sigma^2_ln_Psi = (5/4)(N_f - N_v)^(-7/9)  spectral log-variance
    #   T_M      = 0.8 + sqrt(1.56 / N_s)        97.5th percentile of MAD
    #   FM       = MAD_ln / (T_M * sigma_ln_Psi)
    # Uses natural log (not log10).
    if len(Range) > 1 and len(spec_ratio) > 0:
        N_s = len(spec_ratio)
        mad_ln = np.mean(np.abs(np.log(spec_ratio)))  # natural log
        N_eff = max(num_ffts - n_v, 1)
        sigma_ln = np.sqrt(FM_SIGMA_COEFF * N_eff**FM_SIGMA_EXPONENT)
        T_M = FM_TM_OFFSET + np.sqrt(FM_TM_COEFF / N_s)
        FM = mad_ln / (T_M * sigma_ln)
    else:
        FM = np.nan

    # K_max / K_95: fraction of theoretical spectrum resolved
    K_95 = X_95 * (max(e_4, EPSILON_FLOOR) / nu**3) ** 0.25
    K_max_ratio_val = k_max / K_95 if K_95 > 0 else np.nan

    return e_4, k_max, mad, method, nas_spec, fom, K_max_ratio_val, FM


def _variance_correction(e_3: float, K_upper: float, nu: float, max_iter: int = 50) -> float:
    """Iterative variance correction using Lueck's resolved-variance model.

    Matches ODAS ``get_diss_odas.m`` lines 623-634.  The fraction of total
    Nasmyth variance resolved up to ``K_upper`` is approximated by
    ``tanh(48x) - 2.9x*exp(-22.3x)`` where ``x = (K_upper / ks)^(4/3)``.

    Parameters
    ----------
    e_3 : float
        Uncorrected epsilon [W/kg] from integration up to ``K_upper``.
    K_upper : float
        Upper integration wavenumber [cpm].
    nu : float
        Kinematic viscosity [m²/s].
    max_iter : int
        Maximum iterations (default 50).

    Returns
    -------
    float
        Variance-corrected epsilon [W/kg].
    """
    e_new = e_3
    for _ in range(max_iter):
        x_limit = K_upper * (nu**3 / e_new) ** 0.25
        x_limit = x_limit ** (4 / 3)
        variance_resolved = (
            np.tanh(VARIANCE_TANH_COEFF * x_limit)
            - VARIANCE_EXP_COEFF * x_limit * np.exp(-VARIANCE_EXP_DECAY * x_limit)
        )
        if variance_resolved <= 0:
            break
        e_old = e_new
        e_new = e_3 / variance_resolved
        if e_new / e_old < VARIANCE_CONVERGENCE:
            break
    return e_new


def _inertial_subrange(
    K: np.ndarray, shear_spectrum: np.ndarray, e: float, nu: float, K_limit: float
) -> tuple[float, float]:
    """Fit to the inertial subrange to estimate epsilon.

    Three-pass Nasmyth fit with flyer removal, then two-pass refit.

    Returns
    -------
    epsilon : float
        Dissipation rate [W/kg].
    K_max : float
        Upper fit limit [cpm].
    """
    isr_limit = min(X_ISR * (e / nu**3) ** 0.25, K_limit)
    fit_range = np.where(isr_limit >= K)[0]
    if len(fit_range) < 3:
        fit_range = np.arange(min(3, len(K)))

    k_max = K[fit_range[-1]]

    # Iterative fitting (3 passes)
    for _ in range(3):
        nas = nasmyth_grid(max(e, EPSILON_FLOOR), nu, K[fit_range] + 1e-30)
        ratio = shear_spectrum[fit_range[1:]] / (nas[1:] + 1e-30)
        ratio = ratio[ratio > 0]
        if len(ratio) == 0:
            break
        fit_error = float(np.mean(np.log10(ratio)))
        e = e * 10 ** (3 * fit_error / 2)

    # Remove flyers
    nas = nasmyth_grid(max(e, EPSILON_FLOOR), nu, K[fit_range] + 1e-30)
    if len(fit_range) > 2:
        ratio = shear_spectrum[fit_range[1:]] / (nas[1:] + 1e-30)
        ratio = ratio[ratio > 0]
        if len(ratio) > 0:
            fit_error_vec = np.log10(ratio)
            bad = np.where(np.abs(fit_error_vec) > ISR_FLYER_THRESHOLD)[0]
            if len(bad) > 0:
                bad_limit = max(1, int(np.ceil(ISR_FLYER_FRAC * len(fit_range))))
                if len(bad) > bad_limit:
                    order = np.argsort(fit_error_vec[bad])[::-1]
                    bad = bad[order[:bad_limit]]
                # Remove bad indices from fit_range (offset by 1 for skipped DC)
                keep = np.ones(len(fit_range), dtype=bool)
                for b in bad:
                    if b + 1 < len(keep):
                        keep[b + 1] = False
                fit_range = fit_range[keep]
                k_max = K[fit_range[-1]]

    # Re-fit (2 more passes)
    for _ in range(2):
        nas = nasmyth_grid(max(e, EPSILON_FLOOR), nu, K[fit_range] + 1e-30)
        ratio = shear_spectrum[fit_range[1:]] / (nas[1:] + 1e-30)
        ratio = ratio[ratio > 0]
        if len(ratio) == 0:
            break
        fit_error = float(np.mean(np.log10(ratio)))
        e = e * 10 ** (3 * fit_error / 2)

    return e, k_max


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------


def compute_diss_file(
    source_path: str | Path,
    output_dir: str | Path,
    **diss_kwargs: Any,
) -> list[Path]:
    """Compute epsilon for one file and write NetCDF output(s).

    Parameters
    ----------
    source_path : str or Path
        Input file (.p, full-record .nc, or per-profile .nc).
    output_dir : str or Path
        Output directory for epsilon .nc files.
    **diss_kwargs
        Keyword arguments passed to get_diss.

    Returns
    -------
    list of Path
        Paths to output files written.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = get_diss(source_path, **diss_kwargs)

    output_paths = []
    for i, ds in enumerate(results):
        if len(results) == 1:
            out_name = f"{source_path.stem}_eps.nc"
        else:
            out_name = f"{source_path.stem}_prof{i + 1:03d}_eps.nc"
        out_path = output_dir / out_name
        ds.to_netcdf(out_path)
        output_paths.append(out_path)
        print(
            f"  {out_path.name}: {ds.sizes['time']} estimates, "
            f"P={float(ds.P_mean.min()):.0f}–{float(ds.P_mean.max()):.0f} dbar"  # noqa: RUF001
        )

    return output_paths


def _compute_diss_one(args: tuple) -> tuple[str, int]:
    """Worker for parallel dissipation computation."""
    source_path, output_dir, kwargs = args
    paths = compute_diss_file(source_path, output_dir, **kwargs)
    return str(source_path), len(paths)
