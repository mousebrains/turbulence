# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Per-bead FP07 response equalization from the T1/T2 spectral ratio.

Two FP07 beads on one probe head sample the same temperature microstructure,
so the ratio of their gradient spectra measures their RELATIVE response with
no turbulence model:

    median over windows of ln(S2/S1) = ln G + ln|H2(f)|^2 - ln|H1(f)|^2

The absolute level is not identifiable from in-situ data, so the default FP07
model stays the absolute reference and only the relative part is corrected
(the "geometric mean unity" anchor: e_1 * e_2 = 1, the pair's average response
is exactly the default model):

    |H_i(f)|^2 = |H_model(f; W)|^2 * e_i(f; W)

The correction is applied by EQUALIZING THE DATA: each bead's observed
gradient spectrum and its noise spectrum are divided by e_i before chi is
computed (in full where the bin is signal-dominated, not at all at the noise
floor: :func:`signal_weighted`), mapping the bead onto the pair-average
(default-model) response;
the model, the fit and the variance correction keep the default |H|^2. For a
spectral fit this is identical to multiplying the model by e_i, but the
moment estimators (Method 1, the iterative Method 2) apply the response
through a MODEL-weighted resolved-variance correction, which over-applies a
relative correction whenever the observed spectrum is redder than the model.
On ARCTERX-2022 SN194/SN428 equalizing the data brought ln(chi2/chi1) to
ln G with no K_max_ratio trend left (Q1-Q4 -0.04/+0.03) and halved the
per-window scatter; multiplying the model left most of the trend (-0.25/+0.15).

A correction is measured per (instrument, both channels' bead serials, time
segment): the estimator builds a per-file timeline of the mid-band ratio and
splits it where the pair changed (a swapped probe, drift), and every
correction carries the span it was fitted on. At chi time each file's own
mid-band ratio is checked against its correction (:func:`guard`). See
``docs/fp07_pair_response_PLAN.md`` and the runbook ``docs/fp07pair/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from odas_tpw._version_info import stamp_provenance

# Frequency bands on the REDUCED axis (frequency at the reference speed).
MID_BAND = (15.0, 30.0)  # Hz: the per-file timeline and the chi-time guard
REF_BAND_MAX = 4.0  # Hz: the reference (normalization) level comes from below this
F_LO_MIN = 0.5  # Hz
LOG_STEP = 0.025  # decades per frequency bin of the correction grid
SNR_MIN = 10.0  # both probes, on a running mean of S/N
SNR_SMOOTH = 3  # +/- bins of that running mean
# At chi time the factor is applied in full only where the bin is signal-
# dominated: ln e is weighted by the bin's smoothed S/N, 0 at or below
# APPLY_SNR_LO and 1 at or above SNR_MIN (the S/N the factor was measured at).
# Without this a near-floor window's clipped noise residual max(S-N, 0) is
# scaled by 1/e: on the osu684 glider that inflated the slow bead's chi in the
# low-S/N half of the windows (ln chi2/chi1 +0.36 -> +0.79) while the correction
# was right in the high-S/N windows (-0.39 -> -0.12, toward ln G ~ 0).
APPLY_SNR_LO = 2.0
GUARD_MAX = 0.3  # |ln| between a file's mid-band ratio and its correction
LN_G_MAX = float(np.log(2.0))  # a flat gain beyond this is an amplitude-loss signature
OUTLIER_LNR20 = 0.9  # |ln R'(20 Hz)| beyond the fleet spread: flag a possibly failing bead
F_HI_HALFWIDTH = 0.1  # ln: f_hi is the last bin whose bootstrap half-width is below this
PLACEHOLDER_SERIAL = re.compile(r"^(|\?|t|none|null|na)$", re.IGNORECASE)
ANCHORS = ("geomean", "T1", "T2")
# Flag on a record that covers a refused segment's span: files there match it
# and run UNCORRECTED, saying why, instead of failing to match anything.
NOT_APPLIED = "not_applied"
FORMAT_VERSION = "1"


def is_placeholder(serial: str) -> bool:
    """A bead serial that identifies nothing ('', '?', 'T', ...)."""
    return bool(PLACEHOLDER_SERIAL.match(str(serial).strip()))


def _tau(W: np.ndarray, tau_model: str) -> np.ndarray:
    from odas_tpw.chi.fp07 import fp07_tau

    return np.asarray(fp07_tau(np.maximum(np.asarray(W, dtype=np.float64), 1e-3), model=tau_model))


def reduced_frequency(F: np.ndarray, W: Any, W_ref: float, tau_model: str) -> np.ndarray:
    """Frequency at the reference speed with the same response as F at speed W.

    The response depends on f * tau(W), so F at W corresponds to
    F * tau(W) / tau(W_ref) at W_ref. For the default ``lueck`` model
    (tau ~ W^-1/2) this is F * (W_ref / W)^(1/2); a speed-independent model
    gives F unchanged. Shapes broadcast: W scalar -> (n_f,); W (n_w,) ->
    (n_w, n_f).
    """
    F = np.asarray(F, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    scale = _tau(W, tau_model) / float(_tau(np.array([W_ref]), tau_model)[0])
    if W.ndim == 0:
        return F * float(scale)
    return F[np.newaxis, :] * scale[:, np.newaxis]


@dataclass
class ResponseCorrection:
    """A measured relative response correction for one bead pair and segment."""

    instrument: str
    channels: tuple[str, str]  # gradient channel names, e.g. ("T1_dT1", "T2_dT2")
    serials: tuple[str, str]  # bead serials of those channels
    t_start: float  # epoch seconds: the span this was fitted on (mandatory)
    t_end: float
    f_ref: np.ndarray  # Hz on the reduced axis (frequency at W_ref)
    ln_e: np.ndarray  # (2, n_f): ln e_i, |H_i|^2 = |H_model|^2 * e_i
    f_lo: float
    f_hi: float
    W_ref: float  # m/s
    tau_model: str  # the FP07 tau model the speed scaling follows
    anchor: str = "geomean"
    ln_G: float = 0.0  # flat gain ln(S2/S1) at low f, removed before ln_e
    mid_lnR: float = 0.0  # predicted mid-band ln R' for the chi-time guard
    n_files: int = 0
    n_windows: int = 0
    flags: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)

    # --- application ---------------------------------------------------------

    def factor(self, ci: int, F: np.ndarray, W: Any) -> np.ndarray:
        """e_ci at frequencies F [Hz] and speed(s) W [m/s].

        Below f_lo the factor is held at its f_lo value (~1 by construction:
        the reference level is taken there); above f_hi it is HELD at its
        f_hi value — the ratio is monotone in every pair measured, so tapering
        to 1 would invent a reversal.
        """
        x = reduced_frequency(F, W, self.W_ref, self.tau_model)
        lx = np.log10(np.clip(x, self.f_lo, self.f_hi))
        lf = np.log10(self.f_ref)
        return np.asarray(np.exp(np.interp(lx, lf, self.ln_e[ci])), dtype=np.float64)

    @property
    def applies(self) -> bool:
        """False for a not-applied record (a refused segment's span)."""
        return NOT_APPLIED not in self.flags

    @property
    def reason(self) -> str:
        return str(self.provenance.get("reason", ""))

    def predicted_mid_lnR(self) -> float:
        return self.mid_lnR

    # --- identity ------------------------------------------------------------

    def digest(self) -> str:
        """SHA-256 of the SEMANTIC content: what changes chi, nothing else.

        The file name, location and creation time are excluded, so moving or
        re-timestamping a correction does not re-version chi; refitting it to
        different numbers does.
        """
        payload = {
            "v": FORMAT_VERSION,
            "instrument": self.instrument,
            "channels": list(self.channels),
            "serials": list(self.serials),
            "span": [round(self.t_start, 3), round(self.t_end, 3)],
            "f_ref": np.round(self.f_ref, 6).tolist(),
            "ln_e": np.round(self.ln_e, 6).tolist(),
            "band": [round(self.f_lo, 6), round(self.f_hi, 6)],
            "W_ref": round(self.W_ref, 6),
            "tau_model": self.tau_model,
            "anchor": self.anchor,
            "mid_lnR": round(self.mid_lnR, 6),
            "applies": self.applies,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def matches(
        self, instrument: str, channels: Sequence[str], serials: Sequence[str], t: float
    ) -> tuple[bool, str]:
        """Does this correction describe a file of *instrument* with these beads at time t?

        The instrument must be equal (both sides derive it with
        :func:`pair_identity`). Channels are matched BY NAME, both of them, in
        the correction's order: a swapped channel would apply the correction
        inverted. A real serial must equal the correction's; a placeholder on
        either side ('T', '?') falls back to the instrument and the time span.
        """
        if str(instrument) != self.instrument:
            return False, f"instrument {instrument} is not {self.instrument}"
        ch = list(channels)
        if any(c not in ch for c in self.channels):
            return False, f"channels {ch} do not include {list(self.channels)}"
        sers = {c: str(s) for c, s in zip(channels, serials, strict=False)}
        for c, want in zip(self.channels, self.serials, strict=True):
            have = sers.get(c, "?")
            if not (is_placeholder(want) or is_placeholder(have)) and have != want:
                return False, f"{c} carries bead {have!r}, correction is for {want!r}"
        if not (self.t_start <= t <= self.t_end):
            return False, (
                f"time {_iso(t)} outside the correction's span "
                f"{_iso(self.t_start)}..{_iso(self.t_end)}"
            )
        return True, ""

    # --- persistence ---------------------------------------------------------

    def to_netcdf(self, path: str | Path) -> Path:
        import xarray as xr

        ds = xr.Dataset(
            {
                "ln_e": (("probe", "f"), np.asarray(self.ln_e, dtype=np.float64),
                         {"long_name": "ln of each probe's response relative to the FP07 model",
                          "comment": "applied at speed W by f -> f*tau(W)/tau(W_ref)"}),
                "channel": (("probe",), np.array(self.channels, dtype=object)),
                "serial": (("probe",), np.array(self.serials, dtype=object)),
            },
            coords={"f": ("f", np.asarray(self.f_ref, dtype=np.float64),
                          {"units": "Hz", "long_name": "frequency at the reference speed W_ref"})},
        )
        ds.attrs.update({
            "fp07_pair_format": FORMAT_VERSION,
            "instrument": self.instrument,
            "t_start": self.t_start, "t_end": self.t_end,
            "t_start_iso": _iso(self.t_start), "t_end_iso": _iso(self.t_end),
            "f_lo": self.f_lo, "f_hi": self.f_hi, "W_ref": self.W_ref,
            "tau_model": self.tau_model, "anchor": self.anchor, "ln_G": self.ln_G,
            "mid_lnR": self.mid_lnR, "n_files": self.n_files, "n_windows": self.n_windows,
            "flags": ",".join(self.flags), "digest": self.digest(),
            "convention": "|H_i|^2 = |H_model|^2 * e_i; geomean anchor: e_1*e_2 = 1",
            **{f"prov_{k}": str(v) for k, v in self.provenance.items()},
        })
        stamp_provenance(ds.attrs, "fp07-pair fit")
        path = Path(path)
        ds.to_netcdf(path)
        return path

    @classmethod
    def from_netcdf(cls, path: str | Path) -> ResponseCorrection:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            ds.load()
        a = ds.attrs
        if str(a.get("fp07_pair_format")) != FORMAT_VERSION:
            raise ValueError(
                f"{path}: not an fp07-pair correction (format {a.get('fp07_pair_format')!r})"
            )
        ln_e = np.asarray(ds["ln_e"].values, dtype=np.float64)
        f = np.asarray(ds["f"].values, dtype=np.float64)
        if ln_e.shape != (2, f.size) or not (np.all(np.isfinite(ln_e)) and np.all(f > 0)):
            raise ValueError(f"{path}: ln_e must be finite with shape (2, {f.size})")
        if not np.all(np.diff(f) > 0):
            raise ValueError(f"{path}: f must increase")
        corr = cls(
            instrument=str(a["instrument"]),
            channels=tuple(str(c) for c in ds["channel"].values),  # type: ignore[arg-type]
            serials=tuple(str(s) for s in ds["serial"].values),  # type: ignore[arg-type]
            t_start=float(a["t_start"]), t_end=float(a["t_end"]),
            f_ref=f, ln_e=ln_e, f_lo=float(a["f_lo"]), f_hi=float(a["f_hi"]),
            W_ref=float(a["W_ref"]), tau_model=str(a["tau_model"]),
            anchor=str(a.get("anchor", "geomean")), ln_G=float(a.get("ln_G", 0.0)),
            mid_lnR=float(a.get("mid_lnR", 0.0)), n_files=int(a.get("n_files", 0)),
            n_windows=int(a.get("n_windows", 0)),
            flags=tuple(x for x in str(a.get("flags", "")).split(",") if x),
            provenance={k[5:]: v for k, v in a.items() if k.startswith("prov_")},
        )
        if corr.anchor not in ANCHORS:
            raise ValueError(f"{path}: unknown anchor {corr.anchor!r}")
        if not (corr.t_end > corr.t_start):
            raise ValueError(f"{path}: empty time span")
        stored = a.get("digest")
        if stored and stored != corr.digest():
            raise ValueError(f"{path}: content does not match its stored digest (edited by hand?)")
        return corr


def _iso(t: float) -> str:
    return datetime.fromtimestamp(float(t), UTC).isoformat(timespec="seconds")


_LOAD_CACHE: dict[tuple[str, int, int], ResponseCorrection] = {}


def load_corrections(
    paths: Sequence[str | Path | ResponseCorrection],
) -> list[ResponseCorrection]:
    """Corrections from objects or files, each file read once per process.

    perturb calls this once per profile; the cache (keyed on path, mtime and
    size) keeps a worker from reopening every correction for every profile.
    """
    out: list[ResponseCorrection] = []
    for p in paths:
        if isinstance(p, ResponseCorrection):
            out.append(p)
            continue
        st = os.stat(p)
        key = (str(Path(p).resolve()), st.st_mtime_ns, st.st_size)
        if key not in _LOAD_CACHE:
            _LOAD_CACHE[key] = ResponseCorrection.from_netcdf(p)
        out.append(_LOAD_CACHE[key])
    return out


def select(corrections: Sequence[ResponseCorrection], instrument: str,
           channels: Sequence[str], serials: Sequence[str],
           t: float) -> tuple[ResponseCorrection | None, list[str]]:
    """The one correction matching the file, and the reasons the others did not.

    A correction whose serials EQUAL the file's (as strings, placeholders
    included) beats one that matched only through a placeholder wildcard:
    spans are disjoint within one serial group but their one-hour outer margins
    can reach a neighboring group of the same instrument whose serials changed
    (real <-> placeholder). More than one match of the same rank is an error:
    the result would depend on list order.
    """
    hits: list[ResponseCorrection] = []
    why: list[str] = []
    for c in corrections:
        ok, reason = c.matches(instrument, channels, serials, t)
        if ok:
            hits.append(c)
        else:
            why.append(f"{c.instrument} {c.serials}: {reason}")
    if len(hits) > 1:
        have = {ch: str(s) for ch, s in zip(channels, serials, strict=False)}
        exact = [c for c in hits
                 if all(have.get(ch) == s for ch, s in zip(c.channels, c.serials, strict=True))]
        if len(exact) == 1:
            return exact[0], why
    if len(hits) > 1:
        raise ValueError(
            f"{len(hits)} fp07_response corrections match {list(serials)} at {_iso(t)}; "
            "their time spans overlap"
        )
    return (hits[0] if hits else None), why


# ---------------------------------------------------------------------------
# Per-window log ratio (shared by the estimator and the chi-time guard)
# ---------------------------------------------------------------------------


def _smoothed_snr(S: np.ndarray, N: np.ndarray, k: int = SNR_SMOOTH) -> np.ndarray:
    """Running mean of S/N over +/- k bins along the last axis.

    Normalized by the number of bins actually inside the array: a zero-padded
    mean understated S/N in the first and last k bins (where the reference
    level is taken).
    """
    r = np.where(N > 0, S / np.where(N > 0, N, 1.0), 0.0)
    n = r.shape[-1]
    k = max(0, min(k, (n - 1) // 2))
    ker = np.ones(2 * k + 1)
    count = np.convolve(np.ones(n), ker, mode="same")
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same") / count, -1, r)


def signal_weighted(e: np.ndarray, S: np.ndarray, N: np.ndarray) -> np.ndarray:
    """The factor to apply to one window's bins: e where the signal dominates.

    ln e is scaled by w = clip((s - APPLY_SNR_LO) / (SNR_MIN - APPLY_SNR_LO), 0, 1),
    s the running-mean S/N of the bin (the estimator's gate statistic). The
    factor is a response property of the SIGNAL; in a noise-dominated bin the
    integrand is mostly noise-model error and clipping residual, which the
    bead's response did not shape and which must not be rescaled.
    """
    s = _smoothed_snr(np.asarray(S, dtype=np.float64), np.asarray(N, dtype=np.float64))
    w = np.clip((s - APPLY_SNR_LO) / (SNR_MIN - APPLY_SNR_LO), 0.0, 1.0)
    return np.asarray(np.exp(w * np.log(e)), dtype=np.float64)


def window_log_ratios(S1, S2, N1, N2, snr_min: float = SNR_MIN) -> np.ndarray:
    """ln((S2-N2)/(S1-N1)) per (window, bin); NaN where either probe fails the gate.

    The gate is a running mean of S/N over neighboring bins on BOTH probes,
    not each bin's own noisy value (gating on the estimated quantity biases
    it). Arrays are (window, bin).
    """
    D1, D2 = S1 - N1, S2 - N2
    ok = (_smoothed_snr(S1, N1) >= snr_min) & (_smoothed_snr(S2, N2) >= snr_min)
    ok &= (D1 > 0) & (D2 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, np.log(np.where(ok, D2, 1.0) / np.where(ok, D1, 1.0)), np.nan)


def band_median(lr: np.ndarray, x: np.ndarray, lo: float, hi: float) -> float:
    """Median of finite ln ratios whose reduced frequency lies in [lo, hi]."""
    m = np.isfinite(lr) & (x >= lo) & (x <= hi)
    return float(np.median(lr[m])) if m.sum() >= 10 else np.nan


def guard(corr: ResponseCorrection, lr: np.ndarray, x: np.ndarray) -> tuple[bool, float]:
    """Does a file's own ratio agree with its correction in the mid band?

    *lr* and *x* are (window, bin) log ratios and reduced frequencies for the
    file's windows. Returns (ok, delta) with delta = observed - predicted
    mid-band ln R' (both referenced to the same low band). NaN delta (too few
    usable bins) counts as ok: there is nothing to contradict the correction.
    """
    ref = band_median(lr, x, F_LO_MIN, REF_BAND_MAX)  # the estimator's per-file reference
    mid = band_median(lr, x, *MID_BAND)
    delta = (mid - ref) - corr.mid_lnR
    return (not np.isfinite(delta)) or abs(delta) <= GUARD_MAX, float(delta)


def file_log_ratios(l3_list: Sequence[Any], corr: ResponseCorrection,
                    channel_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(window, bin) log ratios and reduced frequencies over a file's L3 sections.

    The probe order follows the CORRECTION's channels (mapped by name), so the
    ratio is S(corr.channels[1]) / S(corr.channels[0]) whatever order the file
    lists its thermistors in.
    """
    i1, i2 = (list(channel_names).index(c) for c in corr.channels)
    lrs, xs = [], []
    for l3 in l3_list:
        if l3.n_spectra == 0:
            continue
        S, N = l3.gradt_spec, l3.noise_spec  # (probe, freq, window)
        lrs.append(window_log_ratios(S[i1].T, S[i2].T, N[i1].T, N[i2].T))
        xs.append(reduced_frequency(l3.freq, l3.pspd_rel, corr.W_ref, corr.tau_model))
    if not lrs:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return np.vstack(lrs), np.vstack(xs)


def apply_to_l3(l3: Any, corr: ResponseCorrection, channel_names: Sequence[str]) -> None:
    """Attach each probe's factor to *l3*, in place, for ``l4_chi`` to apply.

    ``l3.response_factor[ci]`` gets a callable ``(F, W) -> e``; ``l4_chi``
    divides that probe's observed and noise spectra by it, window by window at
    the window's speed, before computing chi (see the module docstring for why
    the data and not the model). Nothing in *l3* is modified: the stored
    spectra stay raw, so a product can be refitted and its guard recomputed.
    """
    names = list(channel_names)
    factors: list = [None] * len(names)
    for k, ch in enumerate(corr.channels):
        factors[names.index(ch)] = (lambda F, W, _k=k: corr.factor(_k, F, W))
    l3.response_factor = factors


# ---------------------------------------------------------------------------
# Estimation from chi products
# ---------------------------------------------------------------------------


@dataclass
class _FileRatio:
    path: str
    t0: float
    n_windows: int
    med: np.ndarray  # per log-bin median ln ratio (NaN where empty)
    cnt: np.ndarray  # samples per log-bin
    mid: float  # mid-band ln R' (referenced to the file's own low band)


def pair_identity(cfg: dict | None, channels: Sequence[str]) -> tuple[str, list[str]]:
    """(instrument, bead serials of *channels*) from a parsed .p configuration.

    The ONE definition both the estimator and chi use, so a correction is
    keyed and matched on the same strings. Raises when there is no
    configuration: a pair that cannot be identified must neither be pooled
    with others nor matched by time alone.
    """
    from odas_tpw.rsi.p_file import instrument_sn

    if not cfg:
        raise ValueError("no configuration; cannot identify the instrument and beads")
    sn = str(instrument_sn(cfg.get("instrument_info") or {}) or "").strip()
    therm = {str(ch.get("name")): str(ch.get("sn", "")).strip()
             for ch in cfg.get("channels", []) if str(ch.get("type", "")).lower() == "therm"}
    out = []
    for c in channels:
        base = re.sub(r"_d\w+$", "", c)
        out.append(therm.get(base) or therm.get(c) or "?")
    return (f"SN{sn}" if sn else "SN?"), out


def config_from_source(src: str | Path) -> dict:
    """Parsed configuration of a .p file or a NetCDF carrying configuration_string."""
    from odas_tpw.rsi.p_file import parse_config, read_config_string

    src = str(src)
    if src.lower().endswith(".nc"):
        import netCDF4

        with netCDF4.Dataset(src) as s:
            text = getattr(s, "configuration_string", "")
        if not text:
            raise ValueError(f"{src}: no configuration_string; cannot identify the beads")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = read_config_string(src)
    return parse_config(text)


def _locate_source(src: str, product: str | Path) -> str:
    """*src* as written, or, when relative and not found from here, resolved
    against the chi product's ancestors (perturb writes the path relative to
    ITS working directory)."""
    if not src:
        raise ValueError("chi product has no 'source' attribute; cannot identify the beads")
    p = Path(src)
    if p.is_absolute() or p.exists():
        return src
    for anc in Path(product).resolve().parents:
        if (anc / p).exists():
            return str(anc / p)
    raise ValueError(f"source {src} not found (relative to the working directory or to any "
                     f"parent of {product})")


def identity_from_source(src: str, channels: Sequence[str],
                         product: str | Path = ".") -> tuple[str, list[str]]:
    """(instrument, bead serials) of a chi product's source. Raises when unreadable."""
    return pair_identity(config_from_source(_locate_source(src, product)), channels)


def _grid(f_max: float) -> np.ndarray:
    return 10 ** np.arange(np.log10(F_LO_MIN), np.log10(f_max) + 1e-9, LOG_STEP)


def _read_product(path: str) -> dict:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if "spec_gradT" not in ds.variables or ds["spec_gradT"].shape[0] < 2:
            raise ValueError(f"{path}: needs spectra from two thermistors")
        probes = None
        if "probe" in ds.variables:
            pv = ds["probe"][:]
            probes = [str(x) for x in (netCDF4.chartostring(pv) if pv.dtype.kind == "S" else pv)]
        return {
            "S": np.asarray(ds["spec_gradT"][:2], dtype=np.float64),
            "N": np.asarray(ds["spec_noise"][:2], dtype=np.float64),
            "F": np.asarray(ds["F"][:], dtype=np.float64),
            "W": np.asarray(ds["speed"][:], dtype=np.float64),
            "channels": tuple((probes or ["T1_dT1", "T2_dT2"])[:2]),
            "source": str(getattr(ds, "source", "")),
            "start": str(getattr(ds, "start_time", "")),
            "fp07_model": str(getattr(ds, "fp07_model", "single_pole")),
        }


def _epoch(iso: str) -> float:
    if not iso:
        return np.nan
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _file_ratio(
    prod: dict, W_ref: float, tau_model: str, grid: np.ndarray, path: str
) -> _FileRatio:
    S, N, F, W = prod["S"], prod["N"], prod["F"], prod["W"]
    lr = window_log_ratios(S[0].T, S[1].T, N[0].T, N[1].T)  # (window, bin)
    x = np.vstack([reduced_frequency(F[:, j], W[j], W_ref, tau_model) for j in range(W.size)])
    edges = np.sqrt(grid[1:] * grid[:-1])
    b = np.searchsorted(edges, x)
    med = np.full(grid.size, np.nan)
    cnt = np.zeros(grid.size, dtype=int)
    # Nothing below the grid's floor: the DC bin (x = 0) of a detrended
    # spectrum is meaningless and would otherwise land in the lowest bin and
    # become the reference level.
    ok = np.isfinite(lr) & (x >= F_LO_MIN * 0.999) & (x <= grid[-1] * 1.001)
    for k in range(grid.size):
        v = lr[ok & (b == k)]
        cnt[k] = v.size
        if v.size:
            med[k] = np.median(v)
    ref = band_median(lr, x, F_LO_MIN, REF_BAND_MAX)
    mid = band_median(lr, x, *MID_BAND)
    return _FileRatio(path, _epoch(prod["start"]), int(W.size), med, cnt, mid - ref)


def _segments(t: np.ndarray, y: np.ndarray, min_len: int, min_jump: float) -> list[tuple[int, int]]:
    """Split a timeline where its level changes (recursive binary segmentation).

    A split is accepted when the medians of the two sides differ by more than
    *min_jump* AND by more than 4 robust standard errors. Returns half-open
    index ranges into the time-sorted arrays.

    A split is only ever placed between two DIFFERENT times: every profile cut
    from one .p file carries that file's start_time, which is also the key a
    correction is matched on at chi time, so a split inside such a group
    would be an empty span that no file could select (ARCTERX-2022 SN194 has
    groups of 11+ profiles sharing one start).
    """
    out = []

    def rec(a: int, b: int) -> None:
        seg = y[a:b]
        good = np.isfinite(seg)
        if good.sum() < 2 * min_len:
            out.append((a, b))
            return
        best, best_k = 0.0, None
        for k in range(a + min_len, b - min_len + 1):
            if t[k] <= t[k - 1]:
                continue
            left, right = y[a:k], y[k:b]
            left, right = left[np.isfinite(left)], right[np.isfinite(right)]
            if left.size < min_len or right.size < min_len:
                continue
            ml, mr = np.median(left), np.median(right)
            d = abs(ml - mr)
            # Noise from each side's scatter about ITS OWN median: the pooled
            # scatter would include the very step being tested.
            resid = np.concatenate([left - ml, right - mr])
            s = 1.4826 * np.median(np.abs(resid))
            se = s * np.sqrt(1 / left.size + 1 / right.size) if s > 0 else 0.0
            if d > min_jump and d > 4 * se and d > best:
                best, best_k = d, k
        if best_k is None:
            out.append((a, b))
        else:
            rec(a, best_k)
            rec(best_k, b)

    rec(0, len(y))
    return sorted(out)


def reference_bins(ok: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The reference (normalization) bins: the lowest four usable below REF_BAND_MAX.

    Shared with :mod:`odas_tpw.chi.fp07_bead_tau`, so its ratios are referenced
    to exactly the level ``fp07-pair fit`` reports as ``ln G``.
    """
    return np.flatnonzero(ok & (grid <= REF_BAND_MAX))[:4]


def _weighted_median(vals: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(vals) & (w > 0)
    if not m.any():
        return np.nan
    v, ww = vals[m], w[m]
    o = np.argsort(v)
    vs, c = v[o], np.cumsum(ww[o])
    half = 0.5 * c[-1]
    k = int(np.searchsorted(c, half))
    if k + 1 < vs.size and np.isclose(c[k], half, rtol=1e-12, atol=0.0):
        return float(0.5 * (vs[k] + vs[k + 1]))  # an exact half: the two middle values
    return float(vs[k])


def _curve(
    files: list[_FileRatio], grid: np.ndarray, min_windows: int
) -> tuple[np.ndarray, np.ndarray]:
    med = np.array([f.med for f in files])
    cnt = np.array([f.cnt for f in files])
    curve = np.array([_weighted_median(med[:, k], cnt[:, k]) for k in range(grid.size)])
    total = cnt.sum(axis=0)
    curve[total < min_windows] = np.nan
    return curve, total


def _running_median(v: np.ndarray, k: int = 1) -> np.ndarray:
    out = v.copy()
    for i in range(v.size):
        s = v[max(0, i - k): i + k + 1]
        s = s[np.isfinite(s)]
        out[i] = np.median(s) if s.size and np.isfinite(v[i]) else np.nan
    return out


def _running_median_of_estimates(v: np.ndarray, k: int) -> np.ndarray:
    """Running median over the +/- k nearest FINITE entries; NaN stays NaN."""
    out = np.full_like(v, np.nan)
    idx = np.flatnonzero(np.isfinite(v))
    vals = v[idx]
    for n, i in enumerate(idx):
        out[i] = np.median(vals[max(0, n - k): n + k + 1])
    return out


@dataclass
class EstimateResult:
    corrections: list[ResponseCorrection]
    report: list[str]
    refused: list[str]


def estimate(paths: Sequence[str], *, anchor: str = "geomean", min_files: int = 8,
             min_windows: int = 200, segment: bool = True, min_segment_files: int = 10,
             min_jump: float = 0.15, n_boot: int = 100, allow_amplitude: bool = False,
             rng: np.random.Generator | None = None) -> EstimateResult:
    """Corrections for every (instrument, channels, serials) pair in *paths*.

    *paths* are chi product NetCDFs (spec_gradT, spec_noise, F, speed, and a
    'source' attribute naming the per-profile NetCDF or .p file the bead
    serials come from). Every file is accounted for in the report.
    """
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
    from odas_tpw.chi.fp07 import default_tau_model

    rng = rng or np.random.default_rng(0)
    report: list[str] = []
    refused: list[str] = []
    groups: dict[tuple, list[tuple[str, dict]]] = {}
    unreadable = 0
    for p in sorted(paths):
        try:
            prod = _read_product(p)
            inst, ser = identity_from_source(prod["source"], prod["channels"], product=p)
            prod["instrument"], serials = inst, tuple(ser)
        except (OSError, ValueError, KeyError) as exc:
            unreadable += 1
            report.append(f"SKIPPED {os.path.basename(p)}: {exc}")
            continue
        key = (prod["instrument"], prod["channels"], serials, default_tau_model(prod["fp07_model"]))
        groups.setdefault(key, []).append((p, prod))
    report.insert(
        0,
        f"{len(paths)} chi files: {len(paths) - unreadable} read, {unreadable} skipped, "
        f"{len(groups)} bead pair(s)",
    )

    corrections: list[ResponseCorrection] = []
    for (inst, chans, sers, tau_model), members in sorted(groups.items()):
        label = f"{inst} {chans[0]}={sers[0]} {chans[1]}={sers[1]}"
        W_all = np.concatenate([m[1]["W"] for m in members])
        W_ref = float(np.median(W_all[np.isfinite(W_all) & (W_all > 0)]))
        f_top = min(float(np.nanmax(m[1]["F"])) for m in members)
        grid = _grid(min(f_top, 100.0))
        files = [_file_ratio(prod, W_ref, tau_model, grid, p) for p, prod in members]
        files.sort(key=lambda fr: (np.nan_to_num(fr.t0, nan=0.0), fr.path))
        t = np.array([fr.t0 for fr in files])
        y = np.array([fr.mid for fr in files])
        if not np.all(np.isfinite(t)):
            refused.append(f"{label}: some files have no start_time; a time span cannot be set")
            continue
        segs = _segments(t, y, min_segment_files, min_jump) if segment else [(0, len(files))]
        report.append(f"{label}: {len(files)} files, W_ref {W_ref:.3f} m/s, {len(segs)} segment(s)")
        # Spans meet halfway between segments (never overlap: matching must be
        # unambiguous); the outer ends get a one-hour margin.
        bounds = [float(t[0]) - 3600.0]
        for (_a0, b0), (a1, _b1) in pairwise(segs):
            bounds.append(0.5 * (float(t[b0 - 1]) + float(t[a1])))
        bounds.append(float(t[-1]) + 3600.0)
        for si, (a, b) in enumerate(segs):
            sub = files[a:b]
            tag = f"{label} [{_iso(sub[0].t0)}..{_iso(sub[-1].t0)}]"
            t_end = bounds[si + 1] - (1e-3 if si + 1 < len(segs) else 0.0)
            if not t_end > bounds[si]:  # _segments splits only between distinct times
                raise AssertionError(f"{tag}: empty span {bounds[si]} .. {t_end}")
            n_win = int(sum(f.n_windows for f in sub))

            def refuse(why: str, *, _sub=sub, _t0=bounds[si], _t1=t_end, _n=n_win,
                       _tag=tag) -> None:
                # Every span stays covered: its files match a not-applied
                # record and run uncorrected, recording why, rather than
                # failing to match (an error) or silently going uncorrected.
                refused.append(f"{_tag}: {why}")
                corrections.append(ResponseCorrection(
                    instrument=inst, channels=chans, serials=sers, t_start=_t0, t_end=_t1,
                    f_ref=np.array([1.0, 10.0]), ln_e=np.zeros((2, 2)), f_lo=1.0, f_hi=10.0,
                    W_ref=W_ref, tau_model=tau_model, anchor=anchor, n_files=len(_sub),
                    n_windows=_n, flags=(NOT_APPLIED,),
                    provenance={"created": datetime.now(UTC).isoformat(timespec="seconds"),
                                "reason": why,
                                "first_file": os.path.basename(_sub[0].path),
                                "last_file": os.path.basename(_sub[-1].path)}))

            if len(sub) < min_files:
                refuse(f"{len(sub)} files < {min_files}")
                continue
            curve, _total = _curve(sub, grid, min_windows)
            curve = _running_median(curve, 1)
            ok = np.isfinite(curve)
            if ok.sum() < 8:
                refuse("too few usable frequency bins")
                continue
            # Reference: the lowest well-sampled bins below REF_BAND_MAX.
            ref_bins = reference_bins(ok, grid)
            if ref_bins.size == 0:
                refuse(f"no usable bins below {REF_BAND_MAX} Hz for the reference")
                continue
            ln_G = float(np.mean(curve[ref_bins]))
            slope = float(curve[ref_bins[-1]] - curve[ref_bins[0]])
            lnR = curve - ln_G
            # f_hi: last bin, contiguous from the reference, whose bootstrap
            # half-width (over files) is below F_HI_HALFWIDTH.
            boots = []
            for _ in range(n_boot):
                pick = [sub[i] for i in rng.integers(0, len(sub), len(sub))]
                cb, _tot = _curve(pick, grid, min_windows)
                boots.append(_running_median(cb, 1) - np.nanmean(_running_median(cb, 1)[ref_bins]))
            B = np.array(boots)
            with np.errstate(all="ignore"):
                hw = 0.5 * (np.nanpercentile(B, 84, axis=0) - np.nanpercentile(B, 16, axis=0))
            # Smoothed over +/- 2 ESTIMATED bins: with few files the bootstrap
            # of a weighted median is lumpy, and a single spiky bin truncated
            # clean segments at a few Hz (review: 7 of 30 seeds at 12 files).
            # Index neighbors would not do: at low f the log grid is finer than
            # the FFT spacing and most bins are empty.
            hw = _running_median_of_estimates(hw, 2)
            good = ok & np.isfinite(hw) & (hw < F_HI_HALFWIDTH)
            i0 = int(ref_bins[0])
            i1 = i0
            # Contiguous from the reference upward, stepping over bins with NO
            # estimate: at low f the log grid is finer than the FFT spacing, and
            # the speed scaling spreads each FFT bin over several log bins, some
            # of which catch too few windows to estimate. Only a bin that HAS an
            # estimate but is too uncertain stops the band.
            for i in range(i0 + 1, grid.size):
                if not ok[i]:
                    continue
                if not good[i]:
                    break
                i1 = i
            f_lo, f_hi = float(grid[i0]), float(grid[i1])
            if f_hi < MID_BAND[1]:
                refuse(f"usable band ends at {f_hi:.1f} Hz (< {MID_BAND[1]} Hz)")
                continue
            sel = slice(i0, i1 + 1)
            fr, lr = grid[sel], lnR[sel]
            lr = np.interp(np.log10(fr), np.log10(fr[np.isfinite(lr)]), lr[np.isfinite(lr)])
            flags = []
            if abs(ln_G) > LN_G_MAX:
                flags.append("amplitude")
            lr20 = float(np.interp(np.log10(20.0), np.log10(fr), lr))
            if abs(lr20) > OUTLIER_LNR20:
                flags.append("outlier")
            if "amplitude" in flags and not allow_amplitude:
                refuse(f"flat gain ln G = {ln_G:+.2f} (> ln 2): an amplitude-loss signature "
                       "(a failing bead), not a response difference")
                continue
            if anchor == "geomean":
                ln_e = np.vstack([-0.5 * lr, 0.5 * lr])
            elif anchor == "T1":
                ln_e = np.vstack([np.zeros_like(lr), lr])
            else:
                ln_e = np.vstack([-lr, np.zeros_like(lr)])
            # The guard recomputes exactly this per-file statistic at chi time
            # (pooled mid band minus pooled 0.5-4 Hz), so center it on the
            # segment's own files rather than on the smoothed curve.
            mids = np.array([f.mid for f in sub], dtype=float)
            mid = float(np.nanmedian(mids)) if np.isfinite(mids).any() else 0.0
            corr = ResponseCorrection(
                instrument=inst, channels=chans, serials=sers, t_start=bounds[si], t_end=t_end,
                f_ref=fr, ln_e=ln_e, f_lo=f_lo, f_hi=f_hi, W_ref=W_ref, tau_model=tau_model,
                anchor=anchor, ln_G=ln_G, mid_lnR=mid, n_files=len(sub), n_windows=n_win,
                flags=tuple(flags),
                provenance={"created": datetime.now(UTC).isoformat(timespec="seconds"),
                            "estimator": "median over windows of ln(S2/S1), smoothed-SNR gate",
                            "first_file": os.path.basename(sub[0].path),
                            "last_file": os.path.basename(sub[-1].path),
                            "ref_slope": f"{slope:+.3f}"},
            )
            corrections.append(corr)
            spread = np.nanstd([f.mid for f in sub])
            report.append(
                f"  {tag}: {len(sub)} files, {n_win} windows; band {f_lo:.1f}-{f_hi:.1f} Hz; "
                f"ln G {ln_G:+.3f} (low-band slope {slope:+.3f}); ln R' at 20 Hz {lr20:+.3f}; "
                f"per-file mid-band spread {spread:.3f}; flags {','.join(flags) or '-'}")
    return EstimateResult(corrections, report, refused)


def output_name(corr: ResponseCorrection) -> str:
    """File name for *corr*: instrument, bead serials and span start to the minute.

    The start must carry the time, not just the date: a pair that changed
    within a day has two segments starting on one date (seen on a glider).
    """
    def clean(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", s) or "x"
    d0 = datetime.fromtimestamp(corr.t_start, UTC).strftime("%Y%m%dT%H%MZ")
    tail = "" if corr.applies else "_notapplied"
    return (f"fp07pair_{clean(corr.instrument)}_{clean(corr.serials[0])}-{clean(corr.serials[1])}_"
            f"{d0}{tail}.nc")
