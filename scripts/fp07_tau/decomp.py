# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""FP07 response from chi products without imposing a spectral model.

Each window's observed temperature-gradient spectrum, where it is usable, is
modelled as

    ln S(k; w) = a_w + g(log10 k/kB) + h(f)

* ``a_w``  one level per window and probe (chi, plus the constant log-bias of
  a chi-squared spectral estimate at fixed DOF);
* ``g``    the universal spectral shape, one free value per bin of log10(k/kB);
* ``h``    the FP07 response ln|H(f)|^2 -- parametric (single/double pole,
  tau scanned) or one free value per bin of log10 f, pinned to 0 below f0.

A spectral-shape error lives on k/kB and a response error on f = kW; because
kB and W vary between windows, the two separate. See
docs/fp07_tau_prefactor_PLAN.md.

The stored ``spec_gradT`` is the observed spectrum with the first-difference
and bilinear corrections applied and the FP07 response NOT divided out (the
pipeline fits ``spec_batch * H2 + noise`` to it), so ``h`` is the whole
response, not a residual one.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field

import netCDF4
import numpy as np

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


def h2(f: np.ndarray, tau: float, poles: int) -> np.ndarray:
    """|H(f)|^2 for an n-pole response with time constant tau [s]."""
    return (1.0 + (2 * np.pi * f * tau) ** 2) ** (-poles)


def f3db(tau: float, poles: int) -> float:
    """Half-power frequency [Hz] of :func:`h2`."""
    return np.sqrt(2.0 ** (1.0 / poles) - 1.0) / (2 * np.pi * tau)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class Windows:
    """Per (window, probe) spectra and scalars, ragged over frequency."""

    K: list = field(default_factory=list)  # cpm
    F: list = field(default_factory=list)  # Hz
    S: list = field(default_factory=list)  # observed spec_gradT
    N: list = field(default_factory=list)  # noise model
    kB: list = field(default_factory=list)  # cpm, from shear epsilon
    W: list = field(default_factory=list)  # m/s
    chi: list = field(default_factory=list)
    file: list = field(default_factory=list)
    probe: list = field(default_factory=list)
    P: list = field(default_factory=list)
    T: list = field(default_factory=list)
    dof: float = np.nan

    def __len__(self) -> int:
        return len(self.kB)


def load(pattern: str, kB_from: dict | None = None, max_files: int | None = None) -> Windows:
    """Read chi products matching *pattern*.

    *kB_from* maps a chi file path to an array (probe, time) of kB derived
    from the shear epsilon, for products whose own ``kB`` was FITTED
    (chi_method 'fit'); without it the product's ``kB`` is used, which is
    only valid for chi_method 'epsilon'.
    """
    out = Windows()
    files = sorted(glob.glob(pattern))[:max_files]
    for path in files:
        with netCDF4.Dataset(path) as ds:
            method = getattr(ds, "chi_method", "?")
            if kB_from is None and method != "epsilon":
                raise ValueError(f"{path}: chi_method {method!r}; kB was fitted, pass kB_from")
            if np.isnan(out.dof):
                out.dof = float(getattr(ds, "dof_spec", np.nan))
            K = np.asarray(ds["K"][:], float)
            F = np.asarray(ds["F"][:], float)
            S = np.asarray(ds["spec_gradT"][:], float)
            N = np.asarray(ds["spec_noise"][:], float)
            kB = np.asarray(kB_from[path] if kB_from else ds["kB"][:], float)
            W = np.asarray(ds["speed"][:], float)
            chi = np.asarray(ds["chi"][:], float) if "chi" in ds.variables else None
            P = np.asarray(ds["P_mean"][:], float)
            T = np.asarray(ds["T_mean"][:], float)
            n_probe, _, n_t = S.shape
            for p in range(n_probe):
                for j in range(n_t):
                    if not (np.isfinite(kB[p, j]) and kB[p, j] > 0 and np.isfinite(W[j])):
                        continue
                    out.K.append(K[:, j])
                    out.F.append(F[:, j])
                    out.S.append(S[p, :, j])
                    out.N.append(N[p, :, j])
                    out.kB.append(kB[p, j])
                    out.W.append(W[j])
                    out.chi.append(chi[p, j] if chi is not None and chi.ndim == 2 else np.nan)
                    out.file.append(path)
                    out.probe.append(p)
                    out.P.append(P[j])
                    out.T.append(T[j])
    return out


# ---------------------------------------------------------------------------
# Band selection
# ---------------------------------------------------------------------------


@dataclass
class Band:
    snr: float = 3.0  # smoothed S/N must stay above this, from low f up
    f_lo: float = 0.5  # Hz
    f_max: float = 70.0  # Hz (anti-alias filters negligible below)
    k_min: float = 2.0  # cpm (low-k excess)
    smooth: int = 5  # half-width [bins] of the running mean of ln(S/N)


def _contiguous_band(S: np.ndarray, N: np.ndarray, F: np.ndarray, K: np.ndarray, b: Band):
    """Indices of the usable band of one spectrum.

    The upper edge is where a running mean of ln(S/N) first falls below
    ln(snr), not a per-bin test: gating each bin on its own noisy value keeps
    the bins whose estimation noise happened to be positive, which biases the
    high-frequency end -- exactly where the response is measured.
    """
    ok = (b.f_lo <= F) & (b.f_max >= F) & (b.k_min <= K) & (S > 0) & (N > 0)
    ok &= np.isfinite(S) & np.isfinite(N)
    if not ok.any():
        return np.zeros(0, int)
    lr = np.zeros(S.shape)
    lr[ok] = np.log(S[ok] / N[ok])
    kern = np.ones(2 * b.smooth + 1)
    tot = np.convolve(lr, kern, mode="same")
    cnt = np.convolve(ok.astype(float), kern, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        run = np.where(cnt > 0, tot / cnt, -np.inf)  # running mean over usable bins only
    good = ok & (run >= np.log(b.snr))
    idx = np.flatnonzero(ok)
    first = idx[0]
    # extend from the first usable bin while the smoothed SNR holds
    cont = (good | ~ok) & (b.f_max >= F)
    breaks = np.flatnonzero(~cont[first:])
    stop = first + breaks[0] if breaks.size else len(S)
    sel = np.arange(first, stop)
    return sel[ok[sel]]


@dataclass
class Obs:
    y: np.ndarray  # ln S
    lx: np.ndarray  # log10 k/kB
    lf: np.ndarray  # log10 f
    lk: np.ndarray  # log10 k
    f: np.ndarray
    w: np.ndarray  # window index into the Windows it came from
    n_win: int


def select(win: Windows, band: Band, S_override: list | None = None) -> Obs:
    ys, lxs, lfs, lks, fs, ws = [], [], [], [], [], []
    S_all = S_override if S_override is not None else win.S
    for i in range(len(win)):
        S, N, F, K = S_all[i], win.N[i], win.F[i], win.K[i]
        sel = _contiguous_band(S, N, F, K, band)
        if sel.size < 8:
            continue
        ys.append(np.log(S[sel]))
        lxs.append(np.log10(K[sel] / win.kB[i]))
        lfs.append(np.log10(F[sel]))
        lks.append(np.log10(K[sel]))
        fs.append(F[sel])
        ws.append(np.full(sel.size, i))
    cat = np.concatenate
    return Obs(cat(ys), cat(lxs), cat(lfs), cat(lks), cat(fs), cat(ws), len(win))


# ---------------------------------------------------------------------------
# Fixed-effects solvers (alternating group means)
# ---------------------------------------------------------------------------


def _gmean(v: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    s = np.bincount(idx, weights=v, minlength=n)
    c = np.bincount(idx, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, 0.0)


def bins(v: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer bin index of *v* and the bin centers."""
    lo = np.floor(v.min() / width) * width
    idx = np.floor((v - lo) / width).astype(int)
    centers = lo + (np.arange(idx.max() + 1) + 0.5) * width
    return idx, centers


def fit_additive(
    y: np.ndarray,
    groups: list[tuple[np.ndarray, int, np.ndarray | None]],
    n_iter: int = 400,
    tol: float = 1e-9,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Least-squares fit of y = sum_g effect_g[idx_g] by backfitting.

    *groups* is a list of (idx, n_levels, fixed_mask); levels where
    fixed_mask is True are held at 0 (the response's low-f anchor). The first
    group carries the overall level; every other free group is centered.
    """
    effects = [np.zeros(n) for _, n, _ in groups]
    r = y.copy()
    prev = np.inf
    for _ in range(n_iter):
        for gi, (idx, n, fixed) in enumerate(groups):
            r += effects[gi][idx]
            e = _gmean(r, idx, n)
            if fixed is not None:
                e[fixed] = 0.0
            elif gi > 0:
                cnt = np.bincount(idx, minlength=n)
                e -= np.average(e, weights=np.maximum(cnt, 1e-12))
            effects[gi] = e
            r -= e[idx]
        ss = float(r @ r)
        if abs(prev - ss) <= tol * max(ss, 1.0):
            break
        prev = ss
    return effects, r


def scan_tau(obs: Obs, taus: np.ndarray, poles: int, x_width: float = 0.05) -> np.ndarray:
    """Residual sum of squares of the two-way fit at each tau."""
    bx, _ = bins(obs.lx, x_width)
    nx = bx.max() + 1
    ss = np.empty(len(taus))
    for i, tau in enumerate(taus):
        y = obs.y - np.log(h2(obs.f, tau, poles))
        _, r = fit_additive(y, [(obs.w, obs.n_win, None), (bx, nx, None)])
        ss[i] = r @ r
    return ss


def nonparametric_h(
    obs: Obs, f0: float, x_width: float = 0.05, f_width: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three-way fit: window level + g(log x) + h(log f) with h = 0 below f0.

    Returns (f_centers, h, x_centers, g); h is ln|H|^2.
    """
    bx, xc = bins(obs.lx, x_width)
    bf, fc = bins(obs.lf, f_width)
    fixed = 10**fc < f0
    eff, _ = fit_additive(
        obs.y,
        [(obs.w, obs.n_win, None), (bx, bx.max() + 1, None), (bf, bf.max() + 1, fixed)],
    )
    return 10**fc, eff[2], 10**xc, eff[1]


# ---------------------------------------------------------------------------
# Synthetic spectra on a real dataset's geometry
# ---------------------------------------------------------------------------


def kraichnan_shape(x: np.ndarray, q: float = 5.26) -> np.ndarray:
    """Kraichnan gradient spectrum shape vs x = k/kB (arbitrary level)."""
    return x * np.exp(-np.sqrt(6 * q) * x)


def synthesize(
    win: Windows,
    response,  # callable f -> |H|^2
    shape=kraichnan_shape,  # callable x -> relative level
    distort=None,  # callable x -> multiplicative shape error
    dof: float | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Spectra with the real K, F, kB and noise of *win*, a known response,
    a known (optionally distorted) shape, and chi-squared estimation noise.

    Each synthetic window's peak signal-to-noise is set to its real
    counterpart's (max of S/N over 0.5-70 Hz), so the usable band -- and the
    selection effects at its upper edge -- match the data.
    """
    rng = rng or np.random.default_rng(0)
    dof = dof or win.dof
    out = []
    for i in range(len(win)):
        F, N = win.F[i], win.N[i]
        x = win.K[i] / win.kB[i]
        G = shape(x) * (distort(x) if distort else 1.0)
        model = G * response(F)
        band = (F >= 0.5) & (F <= 70) & (N > 0) & np.isfinite(win.S[i])
        with np.errstate(divide="ignore", invalid="ignore"):
            real_peak = np.nanmax(np.where(band, win.S[i] / N, np.nan))
            model_peak = np.nanmax(np.where(band, model / N, np.nan))
        amp = real_peak / model_peak if model_peak > 0 and np.isfinite(real_peak) else 0.0
        true = amp * model + N
        out.append(true * rng.chisquare(dof, size=true.shape) / dof)
    return out


# ---------------------------------------------------------------------------
# kB from the shear epsilon, for products whose own kB was fitted
# ---------------------------------------------------------------------------


def kB_from_diss(chi_files: list[str], diss_dir: str, salinity: float = 35.0) -> dict:
    """{chi file: kB (probe, time)} from the matching diss file's epsilon.

    Uses the geometric mean of the shear probes' epsilon (NaN probes
    skipped), the chi file's nu, and kappa_T(T_mean, S, P_mean). The diss and
    chi windows must coincide (checked to 1 ms).
    """
    import os

    from odas_tpw.scor160.ocean import kappa_T

    out = {}
    for cf in chi_files:
        df = os.path.join(diss_dir, os.path.basename(cf))
        if not os.path.exists(df):
            continue
        with netCDF4.Dataset(cf) as c, netCDF4.Dataset(df) as d:
            tc, td = np.asarray(c["t"][:], float), np.asarray(d["t"][:], float)
            if tc.shape != td.shape or np.nanmax(np.abs(tc - td)) > 1e-3:
                raise ValueError(f"{cf}: chi and diss windows differ")
            eps = np.asarray(d["epsilon"][:], float)  # (probe, time)
            with np.errstate(invalid="ignore", divide="ignore"):
                le = np.where(eps > 0, np.log(eps), np.nan)
                eps_g = np.exp(np.nanmean(le, axis=0))
            nu = np.asarray(c["nu"][:], float)
            kap = kappa_T(np.asarray(c["T_mean"][:], float), salinity, np.asarray(c["P_mean"][:], float))
            kB = (1 / (2 * np.pi)) * (eps_g / (nu * kap**2)) ** 0.25
            n_probe = c["spec_gradT"].shape[0]
            out[cf] = np.tile(kB, (n_probe, 1))
    return out


def thin(win: Windows, n_max: int, key=None, rng=None) -> Windows:
    """At most *n_max* windows, sampled evenly across *key* (default W)."""
    if len(win) <= n_max:
        return win
    rng = rng or np.random.default_rng(0)
    v = np.asarray(key if key is not None else win.W, float)
    order = np.argsort(v)
    keep = np.sort(order[np.linspace(0, len(v) - 1, n_max).astype(int)])
    out = Windows(dof=win.dof)
    for name in ("K", "F", "S", "N", "kB", "W", "chi", "file", "probe", "P", "T"):
        src = getattr(win, name)
        setattr(out, name, [src[i] for i in keep])
    return out
