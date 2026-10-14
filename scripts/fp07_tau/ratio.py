# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Absolute FP07 time constants from the T1/T2 spectral ratio.

Two thermistors a few centimeters apart sample the same turbulence, so their
expected gradient spectra differ only by gain and response:

    <S2(f)> / <S1(f)> = G * |H2(f)|^2 / |H1(f)|^2

The plateau at high f gives tau1/tau2; WHERE the ratio turns over fixes each
tau absolutely. No turbulence spectral model enters (a non-universal shape,
the low-k excess, cancels), and no coherence between the probes is needed.
It needs the two beads to differ: matched beads carry no information about
the absolute level, and the fit reports that rather than a number.

Validated on synthetics in ``ratio_synth.py`` (see
docs/fp07_tau_prefactor_PLAN.md).
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field

import netCDF4
import numpy as np
from scipy.optimize import least_squares

TAU_BOUNDS = (5e-4, 0.2)  # s


def h2(f, tau, poles):
    return (1.0 + (2 * np.pi * f * tau) ** 2) ** (-poles)


def f3db(tau, poles):
    return np.sqrt(2.0 ** (1.0 / poles) - 1.0) / (2 * np.pi * tau)


# ---------------------------------------------------------------------------
# Accumulation: per group, per file, so the bootstrap can resample files
# ---------------------------------------------------------------------------


@dataclass
class Group:
    F: np.ndarray
    d1: dict = field(default_factory=dict)  # file -> sum(S1 - N1)
    d2: dict = field(default_factory=dict)
    ok: dict = field(default_factory=dict)  # file -> count of both-probe SNR passes per bin
    n: dict = field(default_factory=dict)  # file -> windows
    W: list = field(default_factory=list)

    def totals(self, files=None):
        files = list(self.d1) if files is None else files
        return (sum(self.d1[f] for f in files), sum(self.d2[f] for f in files),
                sum(self.ok[f] for f in files), sum(self.n[f] for f in files))


def collect(pattern: str, key, snr_min: float = 10.0, probes=(0, 1), nscale=None) -> dict:
    """Accumulate both probes' noise-subtracted spectra into groups.

    *key(path, ds, j)* returns the group label for window j (e.g. a W bin, or
    (instrument, W bin)), or None to skip it. Windows where either probe is
    non-finite are skipped. One frequency grid per group. *nscale(path, ds)*
    returns per-probe multipliers on the noise model (from a measured floor,
    ``noise_check.py``), default (1, 1).
    """
    groups: dict = {}
    for path in sorted(glob.glob(pattern)):
        try:
            ds = netCDF4.Dataset(path)
        except OSError:
            continue
        with ds:
            if "spec_gradT" not in ds.variables or ds["spec_gradT"].shape[0] <= max(probes):
                continue
            S = np.asarray(ds["spec_gradT"][:], float)
            N = np.asarray(ds["spec_noise"][:], float)
            F = np.asarray(ds["F"][:], float)
            W = np.asarray(ds["speed"][:], float)
            k1, k2 = nscale(path, ds) if nscale else (1.0, 1.0)
            for j in range(S.shape[2]):
                g = key(path, ds, j)
                if g is None:
                    continue
                s1, s2 = S[probes[0], :, j], S[probes[1], :, j]
                n1, n2 = k1 * N[probes[0], :, j], k2 * N[probes[1], :, j]
                if not (np.all(np.isfinite(s1)) and np.all(np.isfinite(s2))):
                    continue
                grp = groups.setdefault(g, Group(F=F[:, j].copy()))
                if grp.F.shape != F[:, j].shape or not np.allclose(grp.F, F[:, j]):
                    continue
                grp.d1[path] = grp.d1.get(path, 0) + (s1 - n1)
                grp.d2[path] = grp.d2.get(path, 0) + (s2 - n2)
                grp.ok[path] = grp.ok.get(path, 0) + ((s1 >= snr_min * n1) & (s2 >= snr_min * n2))
                grp.n[path] = grp.n.get(path, 0) + 1
                grp.W.append(W[j])
    return groups


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _running_median(v, k):
    if k <= 0:
        return v
    return np.array([np.median(v[max(0, i - k): i + k + 1]) for i in range(v.size)])


def fit_ratio(f, ratio, poles):
    """Fit ln R = ln G + ln|H2|^2 - ln|H1|^2. Returns (G, tau1, tau2, rms)."""
    lnR = np.log(ratio)
    lo, hi = np.log(TAU_BOUNDS[0]), np.log(TAU_BOUNDS[1])

    def resid(p):
        lg, lt1, lt2 = p
        return lnR - (lg + np.log(h2(f, np.exp(lt2), poles)) - np.log(h2(f, np.exp(lt1), poles)))

    best = None
    for t1 in (0.002, 0.006, 0.015, 0.04):
        for t2 in (0.002, 0.006, 0.015, 0.04):
            r = least_squares(resid, [0.0, np.log(t1), np.log(t2)], bounds=([-3, lo, lo], [3, hi, hi]))
            if best is None or r.cost < best.cost:
                best = r
    lg, lt1, lt2 = best.x
    return np.exp(lg), np.exp(lt1), np.exp(lt2), float(np.sqrt(2 * best.cost / f.size))


@dataclass
class PoleFit:
    poles: int
    tau1: float
    tau2: float
    G: float
    rms: float
    ci1: tuple = (np.nan, np.nan)
    ci2: tuple = (np.nan, np.nan)
    identifiable: bool = False


@dataclass
class Result:
    label: object
    n_windows: int
    n_files: int
    W: float
    f_band: tuple
    fits: dict  # poles -> PoleFit
    preferred: int | None  # None: the ratio cannot tell the pole form apart


def analyse(label, grp: Group, f_lo=1.0, f_hi=60.0, ok_frac=0.8, median_k=3,
            n_boot=200, rng=None, min_files=8, pole_margin=0.10) -> Result | None:
    """Fit one group with BOTH pole forms; bootstrap over files for intervals.

    A fit is identifiable when both 16-84% intervals lie inside the search
    bounds and each spans less than a factor 1.5. A pole form is preferred
    only if its rms is lower by more than *pole_margin* (fractional): on
    synthetics the ratio often cannot tell the forms apart, and then the
    choice moves f_3dB by 30-40%.
    """
    rng = rng or np.random.default_rng(0)
    files = list(grp.d1)
    d1, d2, ok, n = grp.totals()
    F = grp.F
    band = (ok >= ok_frac * n) & (f_lo <= F) & (f_hi >= F) & (d1 > 0) & (d2 > 0)
    if band.sum() < 12 or len(files) < min_files:
        return None
    f = F[band]

    def one(dd1, dd2, poles):
        rat = _running_median(dd2[band] / dd1[band], median_k)
        if np.any(rat <= 0) or not np.all(np.isfinite(rat)):
            return None
        return fit_ratio(f, rat, poles)

    fits = {}
    picks = [list(rng.choice(files, size=len(files), replace=True)) for _ in range(n_boot)]
    for poles in (1, 2):
        G, t1, t2, rms = one(d1, d2, poles)
        pf = PoleFit(poles, t1, t2, G, rms)
        boots = []
        for pick in picks:
            b1, b2, _, _ = grp.totals(pick)
            r = one(b1, b2, poles)
            if r is not None:
                boots.append(r[1:3])
        if len(boots) >= n_boot // 2:
            b = np.array(boots)
            pf.ci1 = tuple(np.percentile(b[:, 0], [16, 84]))
            pf.ci2 = tuple(np.percentile(b[:, 1], [16, 84]))
            inside = all(TAU_BOUNDS[0] * 1.05 < c < TAU_BOUNDS[1] / 1.05 for c in (*pf.ci1, *pf.ci2))
            narrow = pf.ci1[1] / pf.ci1[0] < 1.5 and pf.ci2[1] / pf.ci2[0] < 1.5
            pf.identifiable = inside and narrow
        fits[poles] = pf
    r1, r2 = fits[1].rms, fits[2].rms
    preferred = 1 if r1 < (1 - pole_margin) * r2 else 2 if r2 < (1 - pole_margin) * r1 else None
    return Result(label, int(n), len(files), float(np.median(grp.W)), (float(f[0]), float(f[-1])),
                  fits, preferred)


def fmt(r: Result) -> str:
    def ms(v):
        return f"{v * 1e3:.1f}"
    lue = 1 / (2 * np.pi * 0.01 * r.W**-0.5)
    head = (f"{r.label!s:>28s}  W {r.W:.2f}  win {r.n_windows:6d} files {r.n_files:4d}  "
            f"{r.f_band[0]:.0f}-{r.f_band[1]:.0f} Hz  pole form: "
            f"{'undetermined' if r.preferred is None else str(r.preferred) + '-pole'}  lueck f3dB {lue:.1f} Hz")
    lines = [head]
    for p, pf in r.fits.items():
        flag = "" if pf.identifiable else "  [NOT IDENTIFIABLE]"
        lines.append(f"{'':30s}{p}-pole rms {pf.rms:.3f}: tau1 {ms(pf.tau1)} [{ms(pf.ci1[0])}-{ms(pf.ci1[1])}]"
                     f"  tau2 {ms(pf.tau2)} [{ms(pf.ci2[0])}-{ms(pf.ci2[1])}] ms  "
                     f"f3dB {f3db(pf.tau1, p):.1f}/{f3db(pf.tau2, p):.1f} Hz{flag}")
    return "\n".join(lines)
