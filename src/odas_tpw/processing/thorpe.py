# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Thorpe-scale overturn analysis for microstructure profiles.

Density inversions (overturns) in a stratified profile carry the signature
of active turbulence.  Adiabatically re-sorting a profile segment to a
statically stable ordering yields, for each sample, the *Thorpe
displacement* — how far the parcel must move to reach its sorted resting
depth (Thorpe 1977):

    ``delta_T(z_j) = z_j - z_sorted(j)``   [m]

The *Thorpe scale* is the rms displacement over the segment,
``L_T = rms(delta_T)``, a proxy for the overturn (energy-containing) size.
Two derived quantities connect it to the dissipation-scale measurements:

- **Patch stratification** ``N2_patch`` — the overturn-weighted background
  stratification the turbulence is working against (Smyth et al. 2001;
  operational form per Kaminski et al. 2021, their Eq. 10):

      ``N2_patch = coef * rms(x - x_sorted) / L_T``

  where ``x`` is the sort proxy and ``coef`` converts its rms fluctuation
  to buoyancy: ``alpha*g`` for temperature, ``g/rho_0`` for potential
  density.  NOTE: the average must be an **rms** — the plain mean of
  ``x_sorted - x`` is identically zero (sorting permutes the same values),
  a trap present in the compact notation of some papers (e.g. Lewin et
  al. 2025, their Eq. 5, whose <.> resolves to Kaminski's rms form).  For
  a fully overturned linear profile ``rms(x - x_sorted) = |dx/dz| * L_T``,
  so ``N2_patch`` recovers the true background ``N2`` exactly.

- **Ozmidov scale** ``L_O = (epsilon / N2_patch^(3/2))^(1/2)`` — the
  largest vertical scale buoyancy allows turbulence to overturn — and the
  ratio ``R_OT = L_O / L_T``, a proxy for the age/state of a mixing event
  (Dillon 1982; Smyth et al. 2001; Mashayek et al. 2021).

Overturn detection is noise-limited: sensor noise produces spurious
cm-scale inversions, so every window carries Galbraith & Kelley (1996)
style diagnostics (same-sign displacement run length, displaced fraction)
and an edge-truncation flag (an overturn clipped by the sort window biases
``L_T`` low).  Callers apply route-dependent ``L_T`` floors — see
``DEFAULT_LT_FLOOR_TEMPERATURE`` / ``DEFAULT_LT_FLOOR_DENSITY``.

References
----------
Thorpe, S.A., 1977: Turbulence and mixing in a Scottish loch.
    Phil. Trans. Roy. Soc. London, A286, 125-181.
    https://doi.org/10.1098/rsta.1977.0112
Dillon, T.M., 1982: Vertical overturns: A comparison of Thorpe and
    Ozmidov length scales. J. Geophys. Res., 87, 9601-9613.
    https://doi.org/10.1029/JC087iC12p09601
Galbraith, P.S. and D.E. Kelley, 1996: Identifying overturns in CTD
    profiles. J. Atmos. Oceanic Technol., 13, 688-702.
    https://doi.org/10.1175/1520-0426(1996)013<0688:IOICP>2.0.CO;2
Smyth, W.D., J.N. Moum, and D.R. Caldwell, 2001: The efficiency of mixing
    in turbulent patches: Inferences from direct simulations and
    microstructure observations. J. Phys. Oceanogr., 31, 1969-1992.
    https://doi.org/10.1175/1520-0485(2001)031<1969:TEOMIT>2.0.CO;2
Kaminski, A.K., E.A. D'Asaro, A.Y. Shcherbina, and R.R. Harcourt, 2021:
    High-resolution observations of the North Pacific transition layer
    from a Lagrangian float. J. Phys. Oceanogr., 51, 3163-3181.
    https://doi.org/10.1175/JPO-D-21-0032.1
Lewin, S.F., A.K. Kaminski, J.M. McSweeney, and A.F. Waterhouse, 2025:
    Multiscale mixing variability on the inner shelf.
    J. Phys. Oceanogr., 55, 1735-1750.
    https://doi.org/10.1175/JPO-D-25-0012.1
"""

from __future__ import annotations

from typing import NamedTuple

import gsw
import numpy as np
import numpy.typing as npt

from odas_tpw.processing.mixing import DEFAULT_MIN_DP

# Sort-window span [s] matching the 4-s segments of Lewin et al. (2025).
# Their span caps detectable overturns at ~2.4-3.2 m at typical VMP fall
# speeds; a narrower window (e.g. a 2-s chi window) halves that cap and
# biases L_T low / R_OT high for large overturns.
DEFAULT_SORT_WINDOW = 4.0

# L_T floors [m] below which a window's overturn signal is considered
# unresolved and callers should fall back to the background N2 (Lewin et
# al. 2025 fall back for L_T < 0.05 m — 18% of their segments).
# - temperature route: 0.05 m (the paper's floor; FP07-class response).
# - density route: provisional 0.10 m — sigma0 from an unpumped JAC CT
#   inherits the thermistor response (~0.1 s => ~7 cm at 0.7 m/s) plus
#   residual salinity spiking; tune from the temperature/density L_T
#   comparison for a given campaign before trusting smaller overturns.
DEFAULT_LT_FLOOR_TEMPERATURE = 0.05
DEFAULT_LT_FLOOR_DENSITY = 0.10

# Minimum samples in a sort window; fewer gives NaN (a handful of points
# cannot distinguish an overturn from noise).
DEFAULT_MIN_SAMPLES = 8

# max|delta_T| above this fraction of the window span raises the
# edge-truncation flag: the overturn plausibly extends past the window,
# so L_T is a lower bound there.
EDGE_TRUNCATION_FRACTION = 0.4

# A displacement at the first/last sample counts as edge truncation only
# when it exceeds this fraction of the span. Requiring merely *nonzero*
# boundary displacement flags essentially every real (noisy) window — a
# one-sample jiggle at the boundary is noise, not a clipped overturn.
EDGE_BOUNDARY_FRACTION = 0.1


class ThorpeDisplacements(NamedTuple):
    """Depth-ordered displacement decomposition of one profile segment."""

    z: np.ndarray  # depth [m, positive down], sorted increasing
    x: np.ndarray  # observed proxy on the depth-ordered grid
    x_sorted: np.ndarray  # stable-sorted (statically stable) proxy profile
    delta: np.ndarray  # Thorpe displacement z_j - z_sorted(j) [m]


class ThorpeStats(NamedTuple):
    """Per-window overturn statistics (all NaN/0 conventions per field)."""

    L_T: float  # rms Thorpe displacement [m]
    rms_fluct: float  # rms proxy fluctuation rms(x - x_sorted)
    frac_displaced: float  # fraction of samples with nonzero displacement
    max_run: int  # longest same-sign run of nonzero displacements
    edge_truncated: bool  # overturn touches window edge / spans too much
    span: float  # vertical span of the window [m]


class WindowThorpeResult(NamedTuple):
    """Arrays of :class:`ThorpeStats` fields, one entry per window."""

    L_T: np.ndarray
    rms_fluct: np.ndarray
    frac_displaced: np.ndarray
    max_run: np.ndarray  # int; 0 where the window is empty/invalid
    edge_truncated: np.ndarray  # bool
    span: np.ndarray
    n: np.ndarray  # samples used per window (int)


def thorpe_displacements(
    z: npt.ArrayLike,
    x: npt.ArrayLike,
    *,
    increasing_down: bool,
) -> ThorpeDisplacements:
    """Thorpe displacements of one segment, sorted to static stability.

    Parameters
    ----------
    z : array_like
        Sample depths [m, positive down]; any order, must be finite.
    x : array_like
        Sort proxy (potential density, or temperature) per sample; finite.
    increasing_down : bool
        Stable ordering of the proxy: ``True`` for density-like proxies
        (densest deepest), ``False`` for temperature in typical ocean
        stratification (warmest shallowest).

    Returns
    -------
    ThorpeDisplacements
        Depth-ordered ``z``, observed ``x``, stable-sorted ``x_sorted``,
        and ``delta = z_j - z_sorted(j)`` (positive when the parcel sits
        deeper than its sorted resting depth).  All sorts are stable, so
        tied values keep their observed order and produce zero
        displacement — a monotonic profile returns ``delta == 0`` exactly.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if z_arr.shape != x_arr.shape or z_arr.ndim != 1:
        raise ValueError("z and x must be 1-D arrays of equal length")
    if z_arr.size < 2:
        raise ValueError("need at least 2 samples to sort a segment")
    if not (np.isfinite(z_arr).all() and np.isfinite(x_arr).all()):
        raise ValueError("z and x must be finite (filter NaN upstream)")

    order = np.argsort(z_arr, kind="stable")
    z_s = z_arr[order]
    x_z = x_arr[order]

    # Stable sort of the proxy into its statically stable ordering. For the
    # descending (temperature) case, negating and sorting ascending keeps
    # the stable tie behavior: equal values stay in observed order.
    key = x_z if increasing_down else -x_z
    val_order = np.argsort(key, kind="stable")
    x_sorted = x_z[val_order]

    # ranks[j]: depth-rank the sample at depth-rank j occupies after sorting.
    ranks = np.empty(z_s.size, dtype=np.intp)
    ranks[val_order] = np.arange(z_s.size)
    delta = z_s - z_s[ranks]

    return ThorpeDisplacements(z=z_s, x=x_z, x_sorted=x_sorted, delta=delta)


def _max_true_run(b: np.ndarray) -> int:
    """Length of the longest run of consecutive True values."""
    if not b.any():
        return 0
    edges = np.diff(np.concatenate(([False], b, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int((ends - starts).max())


def _max_same_sign_run(delta: np.ndarray) -> int:
    """Longest run of consecutive same-sign nonzero displacements.

    The Galbraith & Kelley (1996) run-length idea: uncorrelated sensor
    noise produces sign changes every 1-2 samples, while a real overturn
    displaces a contiguous block of parcels the same way.  Small values
    (~1-2) mean noise; thresholding is left to the caller.
    """
    return max(_max_true_run(delta > 0), _max_true_run(delta < 0))


def thorpe_stats(disp: ThorpeDisplacements) -> ThorpeStats:
    """Summary statistics of one segment's displacement decomposition."""
    delta = disp.delta
    span = float(disp.z[-1] - disp.z[0])
    L_T = float(np.sqrt(np.mean(delta**2)))
    rms_fluct = float(np.sqrt(np.mean((disp.x - disp.x_sorted) ** 2)))
    displaced = delta != 0.0
    frac = float(np.mean(displaced))
    max_run = _max_same_sign_run(delta)
    edge = bool(
        span > 0
        and (
            abs(float(delta[0])) > EDGE_BOUNDARY_FRACTION * span
            or abs(float(delta[-1])) > EDGE_BOUNDARY_FRACTION * span
            or float(np.max(np.abs(delta))) > EDGE_TRUNCATION_FRACTION * span
        )
    )
    return ThorpeStats(
        L_T=L_T,
        rms_fluct=rms_fluct,
        frac_displaced=frac,
        max_run=max_run,
        edge_truncated=edge,
        span=span,
    )


def window_thorpe(
    win_times: npt.ArrayLike,
    win_half_width: float,
    t: npt.ArrayLike,
    P: npt.ArrayLike,
    x: npt.ArrayLike,
    *,
    increasing_down: bool,
    lat: float = 0.0,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_dp: float = DEFAULT_MIN_DP,
) -> WindowThorpeResult:
    """Per-window Thorpe statistics for a cast (window loop like mixing.py).

    For each window (center time ± half width) the samples are depth-sorted
    and stable-sorted by the proxy; windows with fewer than *min_samples*
    finite samples or a pressure span below *min_dp* yield NaN rows
    (``max_run`` 0, ``edge_truncated`` False).

    Parameters
    ----------
    win_times : array_like, shape (n_win,)
        Window center times [s] (same time base as ``t``) — typically the
        chi/dissipation window centers, with ``win_half_width`` from
        ``DEFAULT_SORT_WINDOW`` rather than the (narrower) chi window.
    win_half_width : float
        Half the sort-window duration [s].
    t, P, x : array_like, shape (n,)
        Sample times [s], pressures [dbar], and sort proxy for one cast
        (e.g. slow-grid ``sigma0`` or slow thermistor temperature).
    increasing_down : bool
        See :func:`thorpe_displacements`.
    lat : float
        Latitude for the pressure-to-depth conversion (gsw.z_from_p).
    min_samples, min_dp : int, float
        Validity floors per window.

    Returns
    -------
    WindowThorpeResult
        Per-window ``L_T`` [m], ``rms_fluct`` (proxy units),
        ``frac_displaced``, ``max_run``, ``edge_truncated``, ``span`` [m],
        and ``n`` samples used.
    """
    win_times = np.asarray(win_times, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    # float64 promotion matters: products store sigma0 as float32 whose
    # ~3e-5 kg/m^3 quantization is near the fluctuation scale of small
    # overturns; keep all arithmetic in double precision.
    x = np.asarray(x, dtype=np.float64)
    n_win = len(win_times)

    L_T = np.full(n_win, np.nan)
    rms_fluct = np.full(n_win, np.nan)
    frac = np.full(n_win, np.nan)
    max_run = np.zeros(n_win, dtype=np.intp)
    edge = np.zeros(n_win, dtype=bool)
    span = np.full(n_win, np.nan)
    n_used = np.zeros(n_win, dtype=np.intp)

    depth = -gsw.z_from_p(P, lat)

    for j, tau in enumerate(win_times):
        sel = np.abs(t - tau) <= win_half_width
        if not np.any(sel):
            continue
        Pw = P[sel]
        zw = depth[sel]
        xw = x[sel]
        good = np.isfinite(Pw) & np.isfinite(zw) & np.isfinite(xw)
        if int(np.sum(good)) < min_samples:
            continue
        Pw, zw, xw = Pw[good], zw[good], xw[good]
        if np.ptp(Pw) < min_dp:
            continue
        disp = thorpe_displacements(zw, xw, increasing_down=increasing_down)
        stats = thorpe_stats(disp)
        L_T[j] = stats.L_T
        rms_fluct[j] = stats.rms_fluct
        frac[j] = stats.frac_displaced
        max_run[j] = stats.max_run
        edge[j] = stats.edge_truncated
        span[j] = stats.span
        n_used[j] = len(zw)

    return WindowThorpeResult(
        L_T=L_T,
        rms_fluct=rms_fluct,
        frac_displaced=frac,
        max_run=max_run,
        edge_truncated=edge,
        span=span,
        n=n_used,
    )


def patch_n2(
    rms_fluct: npt.ArrayLike,
    L_T: npt.ArrayLike,
    coef: npt.ArrayLike,
) -> np.ndarray:
    """Overturn-weighted "patch" stratification [s^-2].

    ``N2_patch = coef * rms(x - x_sorted) / L_T`` (Smyth et al. 2001;
    Kaminski et al. 2021, Eq. 10), where ``coef`` converts the proxy's rms
    fluctuation to buoyancy:

    - temperature route: ``coef = alpha * g``  (alpha = thermal expansion
      coefficient [1/K], e.g. ``gsw.alpha`` at the window mean state),
    - density route: ``coef = g / rho_0``  with the proxy ``sigma0``
      [kg/m^3] and ``rho_0 = 1000 + mean(sigma0)``.

    NaN where ``L_T`` is not positive (no resolved overturn: use the
    background N2 instead, per Lewin et al. 2025) or any input is
    non-finite.
    """
    rms_arr = np.asarray(rms_fluct, dtype=np.float64)
    lt_arr = np.asarray(L_T, dtype=np.float64)
    coef_arr = np.asarray(coef, dtype=np.float64)
    ok = (
        np.isfinite(rms_arr)
        & np.isfinite(lt_arr)
        & np.isfinite(coef_arr)
        & (lt_arr > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        n2 = np.where(ok, coef_arr * rms_arr / np.where(ok, lt_arr, 1.0), np.nan)
    return n2


def ozmidov(epsilon: npt.ArrayLike, N2: npt.ArrayLike) -> np.ndarray:
    """Ozmidov scale ``L_O = (epsilon / N^3)^(1/2)`` [m].

    NaN where ``epsilon`` or ``N2`` is non-positive or non-finite (the
    scaling assumes active turbulence in stable stratification).
    """
    eps = np.asarray(epsilon, dtype=np.float64)
    n2 = np.asarray(N2, dtype=np.float64)
    ok = np.isfinite(eps) & np.isfinite(n2) & (eps > 0) & (n2 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(ok, np.sqrt(eps / np.where(ok, n2, 1.0) ** 1.5), np.nan)
    return lo


def r_ot(L_O: npt.ArrayLike, L_T: npt.ArrayLike) -> np.ndarray:
    """Ozmidov-to-Thorpe ratio ``R_OT = L_O / L_T`` [1].

    NaN where ``L_T`` is not positive (no resolved overturn) or ``L_O``
    is non-finite.
    """
    lo = np.asarray(L_O, dtype=np.float64)
    lt = np.asarray(L_T, dtype=np.float64)
    ok = np.isfinite(lo) & np.isfinite(lt) & (lt > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ok, lo / np.where(lt > 0, lt, 1.0), np.nan)
    return ratio


def reynolds_buoyancy(
    epsilon: npt.ArrayLike,
    nu: npt.ArrayLike,
    N2: npt.ArrayLike,
) -> np.ndarray:
    """Buoyancy Reynolds number ``Re_b = epsilon / (nu * N2)`` [1].

    NaN where any input is non-positive or non-finite.
    """
    eps = np.asarray(epsilon, dtype=np.float64)
    nu_arr = np.asarray(nu, dtype=np.float64)
    n2 = np.asarray(N2, dtype=np.float64)
    ok = (
        np.isfinite(eps)
        & np.isfinite(nu_arr)
        & np.isfinite(n2)
        & (eps > 0)
        & (nu_arr > 0)
        & (n2 > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        reb = np.where(ok, eps / (nu_arr * n2), np.nan)
    return reb


def cox_number(
    chi: npt.ArrayLike,
    kappa: npt.ArrayLike,
    dTdz: npt.ArrayLike,
) -> np.ndarray:
    """Cox number ``C_x = chi / (2 * kappa * dTdz^2)`` [1].

    The ratio of turbulent to molecular thermal-variance production
    (Osborn & Cox 1972); Lewin et al. (2025) reject segments with
    ``C_x <= 50`` as too weak for a reliable chi fit.  ``kappa`` is the
    molecular thermal diffusivity (:func:`odas_tpw.scor160.ocean.kappa_T`).
    NaN where ``chi``/``kappa`` is non-positive or ``dTdz`` is zero or any
    input is non-finite.
    """
    chi_arr = np.asarray(chi, dtype=np.float64)
    kappa_arr = np.asarray(kappa, dtype=np.float64)
    grad = np.asarray(dTdz, dtype=np.float64)
    ok = (
        np.isfinite(chi_arr)
        & np.isfinite(kappa_arr)
        & np.isfinite(grad)
        & (chi_arr > 0)
        & (kappa_arr > 0)
        & (grad != 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cx = np.where(ok, chi_arr / (2.0 * kappa_arr * grad**2), np.nan)
    return cx
