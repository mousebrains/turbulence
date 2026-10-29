# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Convenience wrappers for chi computation from PFile / NetCDF sources.

These functions bridge rsi instrument I/O with the pure chi
computation in chi.chi.  They handle data loading, profile
detection, and file-level orchestration.

.. deprecated::
    ``get_chi`` is deprecated.  Use :func:`odas_tpw.rsi.pipeline.run_pipeline`
    or the modular ``process_l2_chi`` → ``process_l3_chi`` → ``process_l4_chi_*``
    chain instead.
"""

from __future__ import annotations

import functools
import logging
import re
import warnings
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from odas_tpw._version_info import stamp_provenance
from odas_tpw.rsi.bad_buffer import UNKNOWN_ATTR
from odas_tpw.rsi.channels import _is_infinity, config_number
from odas_tpw.rsi.dissipation import DOF_NUTTALL

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile

logger = logging.getLogger(__name__)

# Max acceptable epsilon FM (Lueck 2022 MAD-based reject statistic; a good fit
# sits near ~0.7-0.8, NOT 0, ATOMIX rejects FM > ~1.15) for including a probe in the Method-1
# mean. Named for FM, not the variance-ratio `fom` (centered on 1.0) it gates
# only as a fallback when FM is absent. (matches compute_chi_window's default.)
_EPS_FM_LIMIT = 1.15

# Generic RSI differentiator gain, used only when neither the .p config nor
# per-profile NetCDF attributes carry the channel's real value. Real
# instruments differ (SN479: T1_dT1 diff_gain=0.912, T2_dT2 0.920), so the
# fallback is provably wrong there — hence the m8 attr round-trip below.
_DEFAULT_DIFF_GAIN = 0.94

# Exactly the keys _extract_therm_cal can produce (after its g->gain and
# t_0->T_0 renames). extract_profiles writes these (when present in the .p
# config), plus ``diff_gain``, as float variable attrs on each T*_dT*
# per-profile variable, and the NetCDF branch of _load_therm_channels reads
# the same keys back (#131 m8, single source of truth). 'b' is included on
# purpose: noise_thermchannel's eta term consumes it. 'beta_3' comes only with an
# order-3 in-situ fit (#245): the noise model inverts a fitted curve exactly, so
# it needs the whole curve; an embedded beta_3 is never read, so a product on the
# embedded configuration is unchanged.
THERM_CAL_ATTR_KEYS = (
    "e_b", "b", "gain", "beta_1", "beta_2", "beta_3", "adc_fs", "adc_bits", "T_0"
)

# The ``PFile.channel_info`` key under which perturb's in-situ FP07 fit records,
# on the BASE thermistor channel (T1, not T1_dT1), the coefficients that channel's
# data were converted with, in the configuration's form ({t_0, beta_1[, beta_2[,
# beta_3]]}, fp07cal.logr.coeffs_to_config). They replace the embedded t_0/beta_*
# in the noise model (#245). Absent: the data are on the embedded coefficients.
IN_SITU_COEFFS_KEY = "in_situ_coefficients"

# Variable attr naming where T_0/beta_1/beta_2 came from when they are an in-situ
# fit's (its value is the channel's ``calibration`` tag). Written only then, so a
# product on the embedded configuration is unchanged; absent means embedded.
BEAD_COEFFS_SOURCE_ATTR = "bead_coefficients_source"

# Chi-file attr (``<prefix>_<channel>``): windows whose fitted curve had no real
# root at the window's temperature, so the noise model fell back to the order-1
# inversion there (with a warning). Written only when nonzero.
BEAD_INVERSION_FALLBACK_ATTR = "bead_inversion_fallback_windows"


def resolve_tau_scales(
    therm_names: list[str], tau_scale: Mapping[str, float] | None
) -> list[float] | None:
    """Map a ``{channel: multiplier}`` mapping onto the gradient-channel order.

    Accepts either the bead name (``"T1"``) or the gradient channel name
    (``"T1_dT1"``); ``"T1"`` matches ``"T1_dT1"`` because that is how the
    channel is labelled in a .p config and on a calibration sheet, while the
    gradient channel is what chi actually consumes.

    Returns ``None`` for an empty/absent mapping so the caller can pass it
    straight through as "no scaling".

    Raises
    ------
    ValueError
        If a key matches no channel, or matches more than one. Silently
        ignoring an unmatched key would leave a probe uncorrected while the
        config claimed otherwise, and chi goes as the square of the response
        error -- there is no downstream symptom that would reveal it.
    """
    if not tau_scale:
        return None
    scales = [1.0] * len(therm_names)
    for key, value in tau_scale.items():
        hits = [
            i
            for i, name in enumerate(therm_names)
            if name == key or name.startswith(f"{key}_")
        ]
        if len(hits) != 1:
            raise ValueError(
                f"fp07_tau_scale key {key!r} matched {len(hits)} of the "
                f"thermistor channels {therm_names}; expected exactly one"
            )
        v = float(value)
        if not np.isfinite(v) or v <= 0:
            raise ValueError(
                f"fp07_tau_scale[{key!r}] must be finite and > 0, got {value!r}"
            )
        scales[hits[0]] = v
    return scales


def _compute_chi(
    source: PFile | str | Path,
    epsilon_ds: xr.Dataset | None = None,
    fft_length: int = 1024,
    diss_length: int | None = None,
    overlap: int | None = None,
    speed: float | None = None,
    speed_method: str | None = None,
    aoa_deg: float = 3.0,
    direction: str = "auto",
    W_min: float | None = None,
    fp07_model: str = "single_pole",
    bbl_diss_length: int | None = None,
    bbl_overlap: int | None = None,
    bbl_extent: int | None = None,
    anchor: str = "top",
    goodman: bool = True,
    f_AA: float = 98.0,
    fit_method: str = "iterative",
    spectrum_model: str = "kraichnan",
    salinity: npt.ArrayLike | None = None,
    temperature: str | float = "auto",
    conductivity: str = "auto",
    vehicle: str | None = None,
    mask_bad_buffers: bool = True,
    fp07_tau_scale: Mapping[str, float] | None = None,
    fp07_response: Sequence[Any] | None = None,
    fp07_slow_response: Mapping[str, Any] | None = None,
    _pre_loaded: dict[str, Any] | None = None,
) -> list[xr.Dataset]:
    """Compute chi from temperature gradient spectra (internal, no deprecation warning).

    ``speed_method`` selects the through-water speed model (``None`` /
    ``"pressure"`` = historical |dP/dt|; ``"em"`` = U_EM flowmeter;
    ``"flight"`` = inviscid flight model with angle of attack ``aoa_deg``),
    mutually exclusive with a fixed ``speed``. ``W_min`` is the
    profile-detection fall-rate floor [dbar/s]; ``None`` resolves it from
    the vehicle direction (0.3 free-fall, 0.05 glide/horizontal).

    ``temperature`` selects the reference-temperature source for viscosity /
    kappa_T and ``T_mean``: ``"auto"`` (first plausible of T1..Tn, T, JAC_T),
    an explicit channel name (QC failure warns but proceeds), or a number =
    constant reference temperature [degC] (ODAS ``constant_temp`` parity).
    ``conductivity`` selects the channel behind ``salinity="measured"``
    (``"auto"`` = JAC_C when present); ``salinity="measured"`` computes
    practical salinity from the resolved C/T pair and pressure (TEOS-10),
    falling back to ``None`` (visc35) with a warning when no conductivity
    channel exists.

    ``fp07_tau_scale`` maps a thermistor channel name to a multiplier on its
    FP07 time constant, e.g. ``{"T1": 0.30, "T2": 0.75}``. Either the bead name
    (``T1``) or the gradient channel name (``T1_dT1``) is accepted. Unmatched
    keys raise rather than being ignored -- a typo that silently left a probe
    uncorrected would bias chi by the square of the response error with no
    visible symptom. ``None`` (default) leaves every probe on the tau model.

    ``fp07_response`` is a list of measured per-bead response corrections
    (:class:`odas_tpw.chi.fp07_pair.ResponseCorrection` or paths to them, from
    ``fp07-pair fit``). The one matching this file's channels, bead serials
    and start time equalizes each probe's spectra onto the tau model's
    response (geometric-mean anchor: the pair's average stays the tau model).
    No match is an ERROR, never a silent uncorrected product (a refused
    segment's span is covered by a not-applied record instead); a file whose
    own mid-band ratio disagrees with its correction by more than
    ``fp07_pair.GUARD_MAX`` is computed WITHOUT it and says so in its
    attributes. Mutually exclusive with ``fp07_tau_scale``:
    the correction is measured relative to the pair, so stacking it on a
    tau scale would correct the relative part twice.

    ``fp07_slow_response`` maps a bead serial to the slow parallel path of its
    response, ``{T2005: [{beta, tau_s_ms, first, last}, ...]}`` (one mapping or
    a list; :func:`odas_tpw.chi.fp07_slow.parse_slow_response`). A bead found
    on this file's thermistors (read as ``fp07_pair.pair_identity`` reads them)
    whose span contains the file's start_time has ``|H_slow|^2`` divided out of
    its observed and noise spectra; a bead outside every span is recorded and
    left uncorrected; a file without the bead keeps its chi and records only
    its beads (``fp07_slow_response_serials``). The file's own 0.25->2 Hz pair
    ratio is recorded beside the prediction, with a WARNING when they disagree
    by more than ``fp07_slow.SLOW_GUARD_MAX`` over at least
    ``fp07_slow.SLOW_GUARD_MIN_WINDOWS`` windows. Valid only on
    calibrated temperature (the path is relative to the DC level an in-situ
    calibration fixes). Refused with ``fp07_tau_scale`` or ``fp07_response``,
    which were both fitted on spectra that already contain the path.

    ``mask_bad_buffers`` (default True) rejects a chi estimate whose window
    carries an RDL bad-buffer gap (TN-051 s3.2) too long to interpolate on a
    channel that thermistor depends on, or more than
    ``bad_buffer.MAX_INTERP_FRACTION`` of interpolated samples; short gaps are
    repaired at the load boundary regardless. The rejection happens before
    ``chi_final`` is formed, so a contaminated probe cannot be averaged back
    in. ``bad_buffer_fraction`` and ``interpolated_fraction`` record both
    either way.

    ``_pre_loaded`` is a private hook accepting the dict produced by
    :func:`_load_therm_channels` (i.e. channels + ``therm`` + ``diff_gains`` +
    ``therm_cal``) so callers like ``perturb.pipeline`` can avoid the
    redundant NC reads when diss and chi share the same source; when given,
    the dict already carries the resolved T/C/metadata, so
    ``temperature``/``conductivity`` are ignored for loading.
    """
    from odas_tpw.chi.chi import _spectrum_func
    from odas_tpw.chi.fp07 import default_tau_model
    from odas_tpw.chi.l2_chi import process_l2_chi
    from odas_tpw.chi.l3_chi import process_l3_chi
    from odas_tpw.chi.l4_chi import process_l4_chi_epsilon, process_l4_chi_fit
    from odas_tpw.rsi.helpers import (
        _build_l1data_from_channels,
        _validate_speed_selection,
        prepare_profiles,
        resolve_measured_salinity,
    )
    from odas_tpw.scor160.io import L2Params, L3Params
    from odas_tpw.scor160.l2 import process_l2

    if diss_length is None:
        diss_length = 4 * fft_length
    if overlap is None:
        overlap = diss_length // 2
    # Fail fast (before any file access); prepare_profiles re-validates.
    _validate_speed_selection(speed, speed_method)

    # A bead's slow response path: validated before any file access too.
    slow_spans = None
    if fp07_slow_response:
        from odas_tpw.chi.fp07_slow import parse_slow_response

        if fp07_tau_scale or fp07_response:
            raise ValueError(
                "fp07_slow_response is mutually exclusive with fp07_tau_scale and "
                "fp07_response: both were fitted on spectra that already contain the slow "
                "path, so stacking either on it corrects the bead twice"
            )
        slow_spans = parse_slow_response(fp07_slow_response) or None

    # Validate the salinity form BEFORE any file access so a bad string fails
    # fast; "measured" itself is resolved after the channels are loaded.
    salinity_is_measured = False
    if isinstance(salinity, str):
        if salinity.strip().lower() == "measured":
            salinity_is_measured = True
        else:
            # Other string forms (e.g. "hotel:<var>") are resolved upstream by
            # the perturb pipeline; _compute_chi only accepts a concrete value
            # or "measured" here, so fail clearly rather than crash in
            # np.asarray(...).
            raise ValueError(
                f"salinity={salinity!r} is not resolved at this layer; pass a "
                "number, an array, None, or 'measured'. The 'hotel[:<var>]' "
                "option is resolved by the perturb pipeline."
            )

    # Load channels including thermistor data
    data = (
        _pre_loaded
        if _pre_loaded is not None
        else _load_therm_channels(
            source, temperature_name=temperature, conductivity_name=conductivity
        )
    )

    if salinity_is_measured:
        salinity = resolve_measured_salinity(data)

    therm_names = [t[0] for t in data["therm"]]
    n_therm = len(therm_names)
    diff_gains = data.get("diff_gains", [_DEFAULT_DIFF_GAIN] * n_therm)
    therm_cal = data.get("therm_cal", [{}] * n_therm)
    # A fitted bead's curve is inverted exactly by the noise model (#245): a
    # strongly curved per-file fit put the ODAS order-1 inversion more than 2x
    # off in power on 126 of CASPER 2015 doug's 272 fitted channels. Gated on
    # the source, so the embedded configuration keeps the ODAS form bit for bit.
    cal_sources = list(data.get("therm_cal_source") or [None] * n_therm)
    noise_cal = [
        {**cal, "exact_inversion": True} if src else cal
        for cal, src in zip(therm_cal, cal_sources, strict=False)
    ]
    tau_scales = resolve_tau_scales(therm_names, fp07_tau_scale)

    if n_therm == 0:
        raise ValueError("No thermistor gradient channels found")

    pair_corr = None
    if fp07_response:
        pair_corr = _select_pair_correction(
            fp07_response, fp07_tau_scale, data, source, therm_names
        )

    slow_sel = None
    if slow_spans:
        slow_sel = _select_slow_response(slow_spans, data, source, therm_names)

    fs_fast = data["fs_fast"]
    if vehicle is None:
        vehicle = data.get("vehicle", "")

    # Resolve "auto" to a concrete up/down so it doesn't leak into
    # L1Data.profile_dir (prepare_profiles resolves internally but doesn't
    # return it). Mirrors adapter.py / _compute_epsilon.
    from odas_tpw.rsi.vehicle import resolve_direction

    direction = resolve_direction(direction, vehicle)
    # prepare_profiles owns the speed precedence (fixed speed > em/flight
    # method > precomputed speed_fast > pressure) and stamps speed_source/
    # speed_method into data["metadata"] accordingly.
    prepared = prepare_profiles(
        data,
        speed,
        direction,
        salinity,
        vehicle=vehicle,
        W_min=W_min,
        speed_method=speed_method,
        aoa_deg=aoa_deg,
    )
    if prepared is None:
        return []
    (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, _fs_slow, ratio, t_fast) = (
        prepared
    )

    # Pipeline parameters
    l2_params = L2Params(
        HP_cut=0.25,
        despike_sh=np.array([np.inf, 0.5, 0.04]),  # don't despike shear for chi
        despike_A=np.array([np.inf, 0.5, 0.04]),
        profile_min_W=0.05,
        profile_min_P=0.0,
        profile_min_duration=0.0,
        speed_tau=0.0,
    )
    l3_params = L3Params(
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        HP_cut=0.25,
        fs_fast=fs_fast,
        goodman=goodman,
        # "bottom" pins the deepest window's lower edge to the end of the
        # section so the discarded remainder lands at the TOP, not in the
        # boundary layer. Without it a top-anchored 2.0 s window leaves chi's
        # deepest center ~1.2 m above the bed and the BBL has no chi at all.
        anchor=anchor,
        # A shorter window near the bed, exactly as on the shear side: epsilon
        # belongs to a window's CENTER, so halving the window halves the
        # distance from the bed to the deepest estimate.
        bbl_diss_length=bbl_diss_length,
        bbl_overlap=bbl_overlap or 0,
        bbl_extent=bbl_extent,
    )

    grad_func, _ = _spectrum_func(spectrum_model)

    results: list[xr.Dataset] = []
    sections: list[tuple[Any, Any]] = []  # (l2_chi, l3_chi) per profile with spectra
    for s_slow, e_slow in profiles_slow:
        s_fast = s_slow * ratio
        e_fast = min((e_slow + 1) * ratio, len(t_fast))

        sal_prof = sal_fast[s_fast:e_fast] if isinstance(sal_fast, np.ndarray) else sal_fast

        # Build L1Data with temp_fast
        l1 = _build_l1data_from_channels(
            data,
            s_fast,
            e_fast,
            speed_fast,
            P_fast,
            T_fast,
            direction,
            therm_list=data["therm"],
            diff_gains=diff_gains,
            goodman=goodman,
        )

        # L2: section selection and speed
        l2 = process_l2(l1, l2_params)

        # L2_chi: despike temperature, HP-filter vib, compute gradient
        l2_chi = process_l2_chi(l1, l2)

        # L3_chi: temperature gradient spectra
        l3_chi = process_l3_chi(
            l2_chi,
            l3_params,
            fp07_model=fp07_model,
            salinity=sal_prof,
            therm_cal=noise_cal,
            tau_scale=tau_scales,
        )

        if l3_chi.n_spectra == 0:
            continue
        sections.append((l2_chi, l3_chi))

    # A measured per-bead correction is checked against the WHOLE file's own
    # ratio before it is applied (a probe swap or drift since the correction
    # was fitted shows up here), then applied to every section.
    pair_attrs: dict[str, Any] = {}
    pair_vars: dict[str, Any] = {}
    if pair_corr is not None:
        from odas_tpw.chi import fp07_pair

        fname = Path(str(data["metadata"].get("source", ""))).name
        if not pair_corr.applies:
            # A refused segment's record: matched on purpose, applied never.
            ok, delta, reason = False, float("nan"), f"refused at fit: {pair_corr.reason}"
            logger.info("%s: fp07_response not applied (%s)", fname, reason)
        else:
            lr, x = fp07_pair.file_log_ratios([sec[1] for sec in sections], pair_corr,
                                              therm_names)
            ok, delta = fp07_pair.guard(pair_corr, lr, x) if lr.size else (True, float("nan"))
            reason = "" if ok else (
                f"guard: the file's mid-band ln ratio differs from its correction by "
                f"{delta:+.2f} (limit {fp07_pair.GUARD_MAX:.2f}) -- a swapped or drifted pair")
            if ok:
                for _l2c, l3c in sections:
                    fp07_pair.apply_to_l3(l3c, pair_corr, therm_names)
            else:
                logger.warning("%s: fp07_response NOT applied: %s", fname, reason)
        pair_attrs, pair_vars = _pair_provenance(pair_corr, ok, delta, therm_names, reason)

    # A bead's slow path: the guard reads the file's own gradient time series
    # (the chi spectra cannot see below 2 Hz), then every section gets the factor.
    slow_attrs: dict[str, Any] = {}
    if slow_sel is not None:
        slow_attrs = _apply_slow_response(
            slow_sel, slow_spans or {}, sections, therm_names,
            Path(str(data["metadata"].get("source", ""))).name,
        )

    for l2_chi, l3_chi in sections:
        # L4_chi: chi estimation
        if epsilon_ds is not None:
            l4_diss = _epsilon_ds_to_l4data(epsilon_ds)
            l4_chi = process_l4_chi_epsilon(
                l3_chi,
                l4_diss,
                spectrum_model=spectrum_model,
                f_AA=f_AA,
                mask_bad_buffers=mask_bad_buffers,
            )
        else:
            l4_chi = process_l4_chi_fit(
                l3_chi,
                spectrum_model=spectrum_model,
                fit_method=fit_method,
                f_AA=f_AA,
                mask_bad_buffers=mask_bad_buffers,
            )

        # Goodman DOF loss (#131 m9): each coherently-removed vibration signal
        # costs one FFT segment of DOF (Lueck 2022b), mirroring the epsilon
        # dof_spec in dissipation.py.  Gated exactly like l3_chi's do_goodman
        # (goodman requested AND vibration channels present).
        n_v = l2_chi.n_vib if (goodman and l2_chi.n_vib > 0) else 0

        # Build output dataset
        ds = _build_chi_ds_from_pipeline(
            l3_chi,
            l4_chi,
            therm_names,
            fft_length=fft_length,
            diss_length=diss_length,
            overlap=overlap,
            fs_fast=fs_fast,
            fp07_model=fp07_model,
            spectrum_model=spectrum_model,
            fit_method=fit_method,
            f_AA=f_AA,
            grad_func=grad_func,
            chi_method=l4_chi.method,
            n_v=n_v,
            bad_frac=l3_chi.bad_fraction,
            interp_frac=l3_chi.interp_fraction,
        )
        if UNKNOWN_ATTR in data["metadata"]:
            # The source could not say where its bad buffers were (#235).
            for name in ("bad_buffer_fraction", "interpolated_fraction"):
                ds[name] = ds[name].copy(data=np.full(ds[name].shape, np.nan))
        ds.attrs.update(data["metadata"])
        # Per-thermistor FP07 time-constant multipliers, recorded per channel so
        # a chi file is traceable to the tau that produced it without the
        # config. A different tau IS a different chi, and the values are
        # deployment-specific fits rather than anything derivable from the .p
        # file, so without this the number is unrecoverable from the product.
        # Only written when a scale was actually applied, so an untouched file
        # carries no misleading "1.0".
        if tau_scales is not None:
            for name, scale in zip(therm_names, tau_scales, strict=True):
                if scale != 1.0:
                    ds.attrs[f"fp07_tau_scale_{name}"] = float(scale)
            ds.attrs["fp07_tau_model"] = default_tau_model(fp07_model)
        # Which thermistors' noise model ran on an in-situ fit's T_0/beta_*
        # rather than the embedded configuration's (#245). Only those: a file
        # on the embedded coefficients is unchanged.
        for ci, (name, src) in enumerate(zip(therm_names, cal_sources, strict=False)):
            if src:
                ds.attrs[f"{BEAD_COEFFS_SOURCE_ATTR}_{name}"] = src
                n_unsolved = _count_unsolved_inversions(l3_chi.temp, noise_cal[ci])
                if n_unsolved:
                    ds.attrs[f"{BEAD_INVERSION_FALLBACK_ATTR}_{name}"] = n_unsolved
        if pair_attrs:
            ds.attrs.update(pair_attrs)
            ds.attrs["fp07_tau_model"] = default_tau_model(fp07_model)
            for vname, (dims, values, vattrs) in pair_vars.items():
                ds[vname] = (dims, values, vattrs)
        if slow_attrs:
            ds.attrs.update(slow_attrs)
        stamp_provenance(ds.attrs, "chi")
        start_time = data["metadata"].get("start_time", "")
        t_units = f"seconds since {start_time}" if start_time else "seconds"
        ds.coords["t"].attrs.update(
            {
                "standard_name": "time",
                "long_name": "time of chi estimate",
                "units": t_units,
                "calendar": "standard",
                "axis": "T",
            }
        )

        # Cross-probe consistency diagnostic (issue #131): chi_-prefixed
        # per-pair median chi ratio + significance attrs, mirroring the diss
        # build.  sigma_ln uses epsilon_T for the Kolmogorov length (the
        # mk_chi_mean convention).  Computed over ALL finite windows (no fom
        # cut at this stage); the attrs do not survive binning/combo.
        if n_therm >= 2:
            from odas_tpw.processing.probe_consistency import (
                annotate_probe_consistency,
                lueck_ln_sigma,
            )

            sigma_ln = lueck_ln_sigma(
                l4_chi.epsilon_T,
                l3_chi.nu,
                l3_chi.pspd_rel,
                diss_length,
                fs_fast,
                l4_chi.var_resolved,
            )
            src_name = Path(str(data["metadata"].get("source", ""))).name
            annotate_probe_consistency(
                ds,
                l4_chi.chi,
                sigma_ln,
                therm_names,
                quantity="chi",
                attr_prefix="chi_",
                context=f"{src_name} profile {len(results) + 1}",
            )

        results.append(ds)

    return results


def get_chi(
    source: PFile | str | Path,
    epsilon_ds: xr.Dataset | None = None,
    fft_length: int = 1024,
    diss_length: int | None = None,
    overlap: int | None = None,
    speed: float | None = None,
    direction: str = "down",
    fp07_model: str = "single_pole",
    bbl_diss_length: int | None = None,
    bbl_overlap: int | None = None,
    bbl_extent: int | None = None,
    anchor: str = "top",
    goodman: bool = True,
    f_AA: float = 98.0,
    fit_method: str = "iterative",
    spectrum_model: str = "kraichnan",
    salinity: npt.ArrayLike | None = None,
) -> list[xr.Dataset]:
    """Compute chi from temperature gradient spectra.

    .. deprecated::
        Use :func:`odas_tpw.rsi.pipeline.run_pipeline` or the modular
        ``process_l2_chi`` → ``process_l3_chi`` → ``process_l4_chi_*``
        chain instead.

    Parameters
    ----------
    source : PFile, str, or Path
        Input data (same multi-source support as get_diss).
    epsilon_ds : xarray.Dataset or None
        If provided: Method 1 — use known epsilon from shear probes.
        If None: Method 2 — fit Batchelor spectrum for kB.
    fft_length : int
        FFT segment length [samples], default 1024.
    diss_length : int or None
        Dissipation window length [samples], default 4 * fft_length.
    overlap : int or None
        Window overlap [samples], default diss_length // 2.
    speed : float or None
        Fixed profiling speed [m/s]. If None, from dP/dt.
    direction : str
        'up' or 'down' for speed sign convention.
    fp07_model : str
        'single_pole' or 'double_pole' for FP07 transfer function.
    goodman : bool
        Apply Goodman coherent noise removal using accelerometers (default True).
    f_AA : float
        Anti-aliasing filter cutoff [Hz].
    fit_method : str
        Method 2 only: 'mle' (Ruddick et al. 2000) or 'iterative'
        (Peterson & Fer 2014).
    spectrum_model : str
        'batchelor' or 'kraichnan'.
    salinity : float or array_like or None
        Practical salinity [PSU]. If provided, uses gsw-based viscosity
        instead of visc35. Scalar or array matching slow time series.

    Returns
    -------
    list of xarray.Dataset, one per profile, with variables:
        chi         (probe, time) — thermal dissipation rate [K^2/s]
        epsilon_T   (probe, time) — epsilon from T (Method 2) or input epsilon
        kB          (probe, time) — Batchelor wavenumber [cpm]
        K_max_T     (probe, time) — integration limit [cpm]
        fom         (probe, time) — figure of merit (obs/model variance)
        K_max_ratio (probe, time) — K_max / kB (spectral resolution)
        var_resolved (probe, time) — Batchelor gradient-variance resolved
                                     fraction (chi-side V_f; feeds chiLnSigma)
        speed       (time) — profiling speed [m/s]
        nu          (time) — kinematic viscosity [m^2/s]
        T_mean      (time) — mean temperature [deg C]
        P_mean      (time) — mean pressure [dbar]
        spec_gradT  (probe, freq, time) — temperature gradient spectra
        spec_batch  (probe, freq, time) — fitted Batchelor spectra
        spec_noise  (probe, freq, time) — noise spectra (per-probe)
        K           (freq, time) — wavenumber vectors
    """
    warnings.warn(
        "get_chi() is deprecated. Use run_pipeline() or the modular "
        "process_l2_chi → process_l3_chi → process_l4_chi_* chain "
        "instead. See odas_tpw.rsi.pipeline.run_pipeline().",
        DeprecationWarning,
        stacklevel=2,
    )
    return _compute_chi(
        source,
        epsilon_ds=epsilon_ds,
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        speed=speed,
        direction=direction,
        fp07_model=fp07_model,
        goodman=goodman,
        anchor=anchor,
        bbl_diss_length=bbl_diss_length,
        bbl_overlap=bbl_overlap,
        bbl_extent=bbl_extent,
        f_AA=f_AA,
        fit_method=fit_method,
        spectrum_model=spectrum_model,
        salinity=salinity,
    )


# ---------------------------------------------------------------------------
# Pipeline-to-Dataset helpers
# ---------------------------------------------------------------------------


def _epsilon_ds_to_l4data(epsilon_ds: xr.Dataset) -> Any:
    """Convert old-format epsilon xr.Dataset to L4Data for chi computation.

    If the dataset already carries ``epsilonMean`` (written by
    ``processing.epsilon_combine.mk_epsilon_mean``), that QC-filtered geometric
    mean is used as ``epsi_final``.  Otherwise we fall back to a plain
    ``nanmean`` across the per-probe ``epsilon`` array — the historical
    behavior, retained for inputs produced by the lower-level rsi pipeline
    that does not run the multi-probe combine step.

    Time reference: ``L3ChiData.time`` is in "seconds since start_time"
    (small floats). When the diss dataset is read back from NetCDF, xarray
    decodes its ``t`` coord to absolute ``datetime64``. We convert it back
    to seconds-since-``start_time`` here so ``process_l4_chi_epsilon``'s
    nearest-neighbor matching is over a common reference — otherwise every
    chi window collapses onto ``epsilonMean[0]``.
    """
    from odas_tpw.scor160.io import L4Data

    eps_vals = epsilon_ds["epsilon"].values
    if eps_vals.ndim == 2:
        n_sh = eps_vals.shape[0]
    else:
        n_sh = 1
        eps_vals = eps_vals[np.newaxis]

    def _probe_var(name: str) -> np.ndarray | None:
        """Read a per-probe (n_sh, n) variable from the epsilon dataset."""
        if name not in epsilon_ds:
            return None
        a = np.asarray(epsilon_ds[name].values, dtype=np.float64)
        return a[np.newaxis] if a.ndim == 1 else a

    # Probe-rejection filters on the Lueck (2022) MAD-based FM statistic (good
    # a good fit sits near ~0.7-0.8 not 0; ATOMIX rejects FM > ~1.15), NOT the variance-ratio `fom`
    # (centered on 1.0, 0.7-1.4 is a GOOD fit). Thresholding `fom` at 1.15 wrongly
    # drops good probes. Fall back to `fom` only when FM is absent (older files),
    # mirroring l4.py's fom_for_flags.
    fom_arr = _probe_var("FM")
    if fom_arr is None:
        fom_arr = _probe_var("fom")
    mad_arr = _probe_var("mad")

    if "epsilonMean" in epsilon_ds:
        epsi_final = np.asarray(epsilon_ds["epsilonMean"].values, dtype=np.float64)
    elif n_sh > 1:
        # Exclude bad-probe estimates (FM > limit) before averaging across
        # probes, mirroring compute_chi_window's Method-1 filter: a high-FM
        # shear probe must not bias the epsilon that drives every thermistor's
        # chi. Where a window has no good probe, fall back to the all-probe mean.
        eps_for_mean = eps_vals.astype(np.float64, copy=True)
        if fom_arr is not None and fom_arr.shape == eps_for_mean.shape:
            bad = ~np.isfinite(fom_arr) | (fom_arr > _EPS_FM_LIMIT)
            eps_for_mean = np.where(bad, np.nan, eps_for_mean)
        # catch_warnings (not just errstate): an all-NaN window makes nanmean
        # emit a "Mean of empty slice" RuntimeWarning via warnings.warn, which
        # np.errstate does not suppress. Legitimate dropout windows must not
        # produce a warning storm (#9).
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            epsi_final = np.nanmean(eps_for_mean, axis=0)
            allnan = ~np.isfinite(epsi_final)
            if np.any(allnan):
                fallback = np.nanmean(eps_vals, axis=0)
                epsi_final = epsi_final.copy()
                epsi_final[allnan] = fallback[allnan]
    else:
        epsi_final = eps_vals[0]

    times = epsilon_ds.coords["t"].values
    if np.issubdtype(times.dtype, np.datetime64):
        start_str = epsilon_ds.attrs.get("start_time")
        if start_str:
            # HONOR the timezone offset (decode_cf did when it decoded `t`), then
            # normalize to naive UTC — do NOT strip the offset and reinterpret the
            # local wall-clock as the reference, which shifts the epsilon time base
            # by the offset and silently NaNs all chi for a non-UTC instrument.
            _dt = datetime.fromisoformat(start_str)
            if _dt.tzinfo is not None:
                _dt = _dt.astimezone(UTC).replace(tzinfo=None)
            start = np.datetime64(_dt)
        else:
            start = times[0]
        times = ((times - start) / np.timedelta64(1, "ns")).astype(np.float64) / 1e9
    n = len(times)

    return L4Data(
        time=times,
        pres=epsilon_ds["P_mean"].values if "P_mean" in epsilon_ds else np.zeros(n),
        pspd_rel=epsilon_ds["speed"].values if "speed" in epsilon_ds else np.zeros(n),
        section_number=np.ones(n),
        epsi=eps_vals,
        epsi_final=epsi_final,
        epsi_flags=np.zeros((n_sh, n)),
        fom=fom_arr if fom_arr is not None and fom_arr.shape == (n_sh, n)
        else np.zeros((n_sh, n)),
        mad=mad_arr if mad_arr is not None and mad_arr.shape == (n_sh, n)
        else np.zeros((n_sh, n)),
        kmax=np.zeros((n_sh, n)),
        method=np.zeros((n_sh, n)),
        var_resolved=np.zeros((n_sh, n)),
    )


def _reconstruct_spec_batch(l3_chi, l4_chi, grad_func) -> np.ndarray:
    """``(n_gradt, n_wavenumber, n_spec)`` fitted gradient spectra for the
    perturb-diag overlay.

    Bind the per-window molecular thermal diffusivity ``kappa_T`` (mirroring the
    amplitude/fit at ``chi.py:258``) so the overlay matches the stored chi. The
    bare ``grad_func`` would fall back to the module default ``KAPPA_T=1.4e-7``,
    and since the gradient amplitude scales as ``1/kappa_T`` the overlay would
    run ~6% high in warm water where the per-window kappa exceeds the default.
    Falls back to the bare ``grad_func`` only when ``kappa_T`` is not per-window
    sized (e.g. the degenerate no-spectra path). Diagnostic-only — published
    chi/kB/K_T/Gamma come from ``l4_chi`` (already computed with per-window
    kappa), not this reconstruction. (#104 I2a-4.)
    """
    n_spec = l3_chi.n_spectra
    n_gradt = l3_chi.n_gradt
    n_wave = l3_chi.n_wavenumber
    spec_batch = np.full((n_gradt, n_wave, n_spec), np.nan)
    kappa_per_window = l3_chi.kappa_T.size == n_spec
    for j in range(n_spec):
        gf = (
            functools.partial(grad_func, kappa_T=float(l3_chi.kappa_T[j]))
            if kappa_per_window
            else grad_func
        )
        for ci in range(n_gradt):
            kB = l4_chi.kB[ci, j]
            chi_val = l4_chi.chi[ci, j]
            if np.isfinite(kB) and np.isfinite(chi_val):
                spec_batch[ci, :, j] = gf(l3_chi.kcyc[:, j], kB, chi_val)
    return spec_batch


def _build_chi_ds_from_pipeline(
    l3_chi,
    l4_chi,
    therm_names: list[str],
    *,
    fft_length: int,
    diss_length: int,
    overlap: int,
    fs_fast: float,
    fp07_model: str,
    spectrum_model: str,
    fit_method: str,
    f_AA: float,
    grad_func,
    chi_method: str = "epsilon",
    n_v: int = 0,
    bad_frac: np.ndarray | None = None,
    interp_frac: np.ndarray | None = None,
) -> xr.Dataset:
    """Build old-format xr.Dataset from L3ChiData + L4ChiData.

    ``n_v`` is the number of Goodman-removed vibration signals (0 when
    Goodman did not run); each costs one FFT segment of spectral DOF
    (Lueck 2022b), so ``dof_spec`` here matches the epsilon convention in
    ``dissipation.py`` rather than ODAS's uncorrected ``1.9*num_of_ffts``.
    """
    n_spec = l3_chi.n_spectra
    n_freq = l3_chi.n_wavenumber

    # DOF (Nuttall 1971 factor, minus the Goodman DOF loss — see docstring)
    num_ffts = 2 * (diss_length // fft_length) - 1
    dof_spec = DOF_NUTTALL * max(num_ffts - n_v, 1)

    # Reconstruct fitted gradient spectra for the diagnostic overlay.
    spec_batch = _reconstruct_spec_batch(l3_chi, l4_chi, grad_func)

    # F vector (constant across windows) — broadcast view is read-only
    # but suffices for xarray storage and NetCDF serialization.
    F_const = l3_chi.freq
    F_out = np.broadcast_to(F_const[:, np.newaxis], (n_freq, n_spec))

    return _build_chi_dataset(
        chi_out=l4_chi.chi,
        eps_out=l4_chi.epsilon_T,
        kB_out=l4_chi.kB,
        K_max_out=l4_chi.K_max,
        fom_out=l4_chi.fom,
        # getattr: an L4ChiData-like object from before mad_shape still builds.
        mad_shape_out=getattr(l4_chi, "mad_shape", None),
        K_max_ratio_out=l4_chi.K_max_ratio,
        var_resolved_out=l4_chi.var_resolved,
        speed_out=l3_chi.pspd_rel,
        nu_out=l3_chi.nu,
        P_out=l3_chi.pres,
        T_out=l3_chi.temp,
        t_out=l3_chi.time,
        spec_gradT=l3_chi.gradt_spec,
        spec_batch=spec_batch,
        spec_noise_out=l3_chi.noise_spec,
        K_out=l3_chi.kcyc,
        F_out=F_out,
        therm_names=therm_names,
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        fs_fast=fs_fast,
        dof_spec=dof_spec,
        fp07_model=fp07_model,
        spectrum_model=spectrum_model,
        fit_method=fit_method,
        f_AA=f_AA,
        chi_method=chi_method,
        bad_frac_out=(
            bad_frac
            if bad_frac is not None and bad_frac.shape == l4_chi.chi.shape
            else np.zeros_like(l4_chi.chi)
        ),
        interp_frac_out=(
            interp_frac
            if interp_frac is not None and interp_frac.shape == l4_chi.chi.shape
            else np.zeros_like(l4_chi.chi)
        ),
    )


def _build_chi_dataset(
    *,
    chi_out: np.ndarray,
    eps_out: np.ndarray,
    kB_out: np.ndarray,
    K_max_out: np.ndarray,
    fom_out: np.ndarray,
    K_max_ratio_out: np.ndarray,
    var_resolved_out: np.ndarray,
    bad_frac_out: np.ndarray,
    interp_frac_out: np.ndarray,
    speed_out: np.ndarray,
    nu_out: np.ndarray,
    P_out: np.ndarray,
    T_out: np.ndarray,
    t_out: np.ndarray,
    spec_gradT: np.ndarray,
    spec_batch: np.ndarray,
    spec_noise_out: np.ndarray,
    K_out: np.ndarray,
    F_out: np.ndarray,
    therm_names: list[str],
    fft_length: int,
    diss_length: int,
    overlap: int,
    fs_fast: float,
    dof_spec: float,
    fp07_model: str,
    spectrum_model: str,
    fit_method: str,
    f_AA: float,
    chi_method: str = "epsilon",
    mad_shape_out: np.ndarray | None = None,
) -> xr.Dataset:
    """Build an xarray Dataset from chi estimation output arrays.

    ``mad_shape_out`` defaults to all-NaN (same shape as ``fom_out``) for a
    caller with no shape statistic to report.
    """
    from odas_tpw.rsi.helpers import _build_result_dataset

    if mad_shape_out is None:
        mad_shape_out = np.full(np.shape(fom_out), np.nan)
    elif np.shape(mad_shape_out) != np.shape(fom_out):
        raise ValueError(
            f"mad_shape shape {np.shape(mad_shape_out)} != fom shape {np.shape(fom_out)}"
        )

    variables = [
        (
            "chi",
            ["probe", "time"],
            chi_out,
            {
                "units": "K2 s-1",
                "long_name": "thermal variance dissipation rate",
            },
        ),
        (
            "epsilon_T",
            ["probe", "time"],
            eps_out,
            {
                "units": "W kg-1",
                "long_name": "TKE dissipation rate from temperature",
            },
        ),
        (
            "kB",
            ["probe", "time"],
            kB_out,
            {
                "units": "cpm",
                "long_name": "Batchelor wavenumber",
            },
        ),
        (
            "K_max_T",
            ["probe", "time"],
            K_max_out,
            {
                "units": "cpm",
                "long_name": "upper wavenumber integration limit for chi",
            },
        ),
        (
            "fom",
            ["probe", "time"],
            fom_out,
            {
                "units": "1",
                "long_name": "figure of merit (observed/model variance ratio)",
                "comment": (
                    "Ratio of observed to attenuated-model variance over the fit "
                    "range; values near 1.0 indicate a good fit. NOT the MAD-based "
                    "ATOMIX/Rockland FM statistic (Lueck 2022), for which a good "
                    "fit sits near ~0.7-0.8 (NOT 0) and the recommended QC limit is ~1.15."
                ),
            },
        ),
        (
            "mad_shape",
            ["probe", "time"],
            mad_shape_out,
            {
                "units": "1",
                "long_name": "spread of log10(observed/model) across the chi fit band",
                "comment": (
                    "mean_k |r_k - median(r)| in dex, r_k = log10(observed / model) per "
                    "wavenumber bin, over the same bins and the same model as fom "
                    "(Batchelor/Kraichnan x FP07 |H|^2 + noise). A diagnostic of SHAPE, "
                    "not of level: centering on the window's own median log ratio "
                    "removes any constant offset (that is fom's job) and the bias of a "
                    "chi-squared spectral estimate, whose median and mean log sit below "
                    "its mean. fom is an integral, so a model wrong by ~2x bin by bin "
                    "can still give fom ~1 (issue #191); this cannot cancel. A perfect "
                    "fit is NOT 0: spectral scatter floors it at ~0.17 / 0.14 / 0.11 dex "
                    "for 9.5 / 13.3 / 19 degrees of freedom "
                    "(odas_tpw.chi.chi.mad_shape_floor). Use the WINDOW's dof: the "
                    "dof_spec attribute is for diss_length, and near-bed windows laid "
                    "out with bbl_diss_length average fewer FFT segments, so their "
                    "floor is higher (dof = 1.9 * (2 * bbl_diss_length // fft_length "
                    "- 1 - n_v), n_v = Goodman-removed signals). The epsilon product's "
                    "mad is the same statistic taken about 0. Not a QC gate."
                ),
            },
        ),
        (
            "K_max_ratio",
            ["probe", "time"],
            K_max_ratio_out,
            {
                "units": "1",
                "long_name": "K_max / kB spectral resolution ratio",
            },
        ),
        (
            "var_resolved",
            ["probe", "time"],
            var_resolved_out,
            {
                "units": "1",
                "long_name": "Batchelor gradient-spectrum variance-resolved fraction",
                "comment": (
                    "Fraction of the model temperature-gradient (Batchelor/"
                    "Kraichnan) variance within the fit band [K_min, K_max], "
                    "relative to the full spectrum; NO FP07 |H|^2 attenuation "
                    "(the chi-side analog of the Nasmyth V_f, Lueck 2022 eq 17). "
                    "mk_chi_mean applies the eq (18) truncation "
                    "L_hat_f = L_hat * var_resolved**0.75 to widen chiLnSigma "
                    "where the spectrum is truncated at K_max. (issue #104 U4-F1.) "
                    "NOTE: the diss (epsilon) product also has a 'var_resolved' "
                    "but it is the Nasmyth SHEAR fraction; the two are distinct "
                    "per-product quantities and must not be cross-read (mk_chi_mean "
                    "reads only the chi product, mk_epsilon_mean only the diss product)."
                ),
            },
        ),
        (
            "bad_buffer_fraction",
            ["probe", "time"],
            bad_frac_out,
            {
                "units": "1",
                "long_name": "fraction of window dropped as unrepairable RDL bad buffer",
                "comment": (
                    "RDL v6.1+ substitutes a sentinel for missing samples "
                    "(TN-051 rev. 2026-01-12 s3.2). This counts only the gaps too "
                    "long to interpolate through (> 0.25 s), per thermistor over "
                    "the channels THAT thermistor depends on: its own gradient "
                    "channel and base, the vibration stack when Goodman cleaning "
                    "is on, pressure, the reference T/C, and the speed inputs "
                    "actually used — a U_EM dropout counts only when the speed "
                    "came from U_EM, not under a flight model. Nonzero values "
                    "have chi, epsilon_T and var_resolved set to NaN (before "
                    "chi_final is formed) unless mask_bad_buffers=False."
                ),
            },
        ),
        (
            "interpolated_fraction",
            ["probe", "time"],
            interp_frac_out,
            {
                "units": "1",
                "long_name": "fraction of window linearly interpolated across short RDL gaps",
                "comment": (
                    "RDL bad-buffer gaps of 0.25 s or less are repaired by linear "
                    "interpolation before the spectra rather than rejected. chi is "
                    "still set to NaN where this exceeds 0.05, bounding the "
                    "variance lost to the repair."
                ),
            },
        ),
        (
            "speed",
            ["time"],
            speed_out,
            {
                "units": "m s-1",
                "long_name": "profiling speed",
            },
        ),
        (
            "nu",
            ["time"],
            nu_out,
            {
                "units": "m2 s-1",
                "long_name": "kinematic viscosity of sea water",
            },
        ),
        (
            "P_mean",
            ["time"],
            P_out,
            {
                "units": "dbar",
                "long_name": "mean sea water pressure",
                "standard_name": "sea_water_pressure",
                "positive": "down",
            },
        ),
        (
            "T_mean",
            ["time"],
            T_out,
            {
                "units": "degree_Celsius",
                "long_name": "mean sea water temperature",
                "standard_name": "sea_water_temperature",
            },
        ),
        (
            "spec_gradT",
            ["probe", "freq", "time"],
            spec_gradT,
            {
                "units": "K2 m-1",
                "long_name": "temperature gradient wavenumber spectrum (observed)",
            },
        ),
        (
            "spec_batch",
            ["probe", "freq", "time"],
            spec_batch,
            {
                "units": "K2 m-1",
                "long_name": "fitted Batchelor temperature gradient spectrum",
            },
        ),
        (
            "spec_noise",
            ["probe", "freq", "time"],
            spec_noise_out,
            {
                "units": "K2 m-1",
                "long_name": "FP07 electronics noise spectrum",
            },
        ),
        (
            "K",
            ["freq", "time"],
            K_out,
            {
                "units": "cpm",
                "long_name": "wavenumber (cycles per meter)",
            },
        ),
        (
            "F",
            ["freq", "time"],
            F_out,
            {
                "units": "Hz",
                "long_name": "frequency",
            },
        ),
    ]
    return _build_result_dataset(
        variables,
        therm_names,
        t_out,
        "thermistor probe name",
        {
            "Conventions": "CF-1.13, ACDD-1.3",
            "fft_length": fft_length,
            "diss_length": diss_length,
            "overlap": overlap,
            "fs_fast": fs_fast,
            "dof_spec": dof_spec,
            "fp07_model": fp07_model,
            "spectrum_model": spectrum_model,
            "fit_method": fit_method,
            "f_AA": f_AA,
            "chi_method": chi_method,
        },
    )


# ---------------------------------------------------------------------------
# Channel loading helpers
# ---------------------------------------------------------------------------


def _extract_therm_cal(
    ch_cfg: dict[str, Any], fitted: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Extract thermistor calibration parameters from PFile channel config.

    Config keys are lowercase (parse_config lowercases all keys); 'g' and
    't_0' are renamed to the 'gain'/'T_0' parameter names expected by
    noise_thermchannel.

    Values are read by :func:`~odas_tpw.rsi.channels.config_number`, the
    converters' own parser, so the noise model and ``convert_therm`` cannot
    disagree about an arithmetic coefficient (``t_0 = 273.15+16``).  The
    linear term follows ``convert_therm``'s precedence (legacy ``beta`` before
    ``beta_1``), and ``beta_2 = Inf`` is no quadratic term, as there.

    *fitted* is an in-situ fit's ``{t_0, beta_1[, beta_2[, beta_3]]}``
    (:data:`IN_SITU_COEFFS_KEY`): the bead's R(T), which the channel's data
    were converted with, so it replaces the embedded ``T_0``/``beta_1``/
    ``beta_2`` WHOLE and adds ``beta_3`` when it has one. An order-1 fit has
    no quadratic term, so an embedded ``beta_2`` must not survive under it.
    ``e_b``, ``b``, ``gain`` and ``adc_*`` describe the electronics and stay
    the embedded values. Without a fit ``beta_3`` is never read: the noise
    model's ODAS form has no cubic term (an ``fp07-cal`` patched order-3
    configuration keeps that limitation), and a fit's curve is inverted
    exactly instead (:func:`odas_tpw.chi.fp07.bead_inversion`).
    """
    cal = {}
    name = str(ch_cfg.get("name", "?"))
    beta_key = "beta" if ch_cfg.get("beta") not in (None, "") else "beta_1"
    for key in ("e_b", "b", "g", beta_key, "beta_2", "adc_fs", "adc_bits", "t_0"):
        val = ch_cfg.get(key)
        if val in (None, "") or (key == "beta_2" and _is_infinity(val)):
            continue
        cal["beta_1" if key == "beta" else key] = config_number(val, key=key, channel=name)
    # Map config keys to noise_thermchannel parameter names
    if "g" in cal:
        cal["gain"] = cal.pop("g")
    if "t_0" in cal:
        cal["T_0"] = cal.pop("t_0")
    if fitted is not None:
        cal["T_0"] = float(fitted["t_0"])
        cal["beta_1"] = float(fitted["beta_1"])
        if fitted.get("beta_2") is not None:
            cal["beta_2"] = float(fitted["beta_2"])
        else:
            cal.pop("beta_2", None)
        if fitted.get("beta_3") is not None:
            cal["beta_3"] = float(fitted["beta_3"])
    return cal


def _base_therm_name(name: str) -> str:
    """The bead's own channel: ``T1`` for ``T1_dT1``, *name* itself otherwise."""
    m = re.match(r"^(\w+)_d\1$", name)
    return m.group(1) if m else name


def _bead_fit(
    channel_info: Mapping[str, Mapping[str, Any]] | None, name: str
) -> Mapping[str, float] | None:
    """The in-situ fit recorded for *name*'s bead, or None (embedded coefficients).

    *name* is a thermistor channel (``T1``) or its pre-emphasized gradient
    channel (``T1_dT1``); the fit is recorded on the base channel.
    """
    if not channel_info:
        return None
    fit = channel_info.get(_base_therm_name(name), {}).get(IN_SITU_COEFFS_KEY)
    return fit if fit else None


def _bead_coeffs_source(
    channel_info: Mapping[str, Mapping[str, Any]] | None, name: str
) -> str | None:
    """:data:`BEAD_COEFFS_SOURCE_ATTR`'s value for *name*, or None (embedded).

    The bead's ``calibration`` tag, which names the fit's order and reference.
    """
    if _bead_fit(channel_info, name) is None:
        return None
    assert channel_info is not None  # _bead_fit returned a fit
    tag = channel_info.get(_base_therm_name(name), {}).get("calibration")
    return str(tag) if tag else "in-situ fit"


def _count_unsolved_inversions(T_means: Any, cal: Mapping[str, Any]) -> int:
    """Windows the noise model could not invert exactly (no real root).

    Zero unless *cal* asks for the exact inversion of a curved polynomial;
    a non-finite temperature is not counted (it gives no noise model either way).
    """
    if not cal.get("exact_inversion") or ("beta_2" not in cal and "beta_3" not in cal):
        return 0
    from odas_tpw.chi.fp07 import bead_inversion

    T = np.asarray(T_means, dtype=np.float64)
    if T.size == 0:
        return 0
    _, _, solved = bead_inversion(
        T, cal["T_0"], cal["beta_1"], cal.get("beta_2"), cal.get("beta_3")
    )
    return int(np.count_nonzero(~solved & np.isfinite(T)))


def _therm_gradient_config(
    config: dict[str, Any],
    name: str,
    channel_info: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[float, dict[str, float]]:
    """``(diff_gain, therm_cal)`` for a pre-emphasized T*_dT* channel.

    ``diff_gain`` comes from the gradient channel's own config section; the
    thermistor calibration comes from the base channel (T1 for T1_dT1) via
    :func:`_extract_therm_cal`, with the bead's in-situ fit from
    *channel_info* (``PFile.channel_info``) in place of the embedded
    ``T_0``/``beta_*`` when perturb fitted one (#245). Single source of truth
    (#131 m8 F7): the .p branch of :func:`_load_therm_channels` and the
    per-profile NetCDF write path (``profile._load_from_pfile``) both call
    this, so what ``extract_profiles`` embeds as variable attrs is
    byte-for-byte what the .p path computes.
    """
    ch_cfg: dict = next(
        (ch for ch in config["channels"] if ch.get("name") == name),
        {},
    )
    diff_gain = config_number(
        ch_cfg.get("diff_gain"), _DEFAULT_DIFF_GAIN, key="diff_gain", channel=name
    )
    base_name = _base_therm_name(name)
    base_cfg: dict = next(
        (ch for ch in config["channels"] if ch.get("name") == base_name),
        {},
    )
    return diff_gain, _extract_therm_cal(base_cfg, _bead_fit(channel_info, base_name))


def _pair_identity(source: Any, therm_names: list[str]) -> tuple[str, list[str]]:
    """(instrument, bead serials) of this file, by fp07_pair's one definition.

    Read from the .p header (no data records) or the NetCDF's
    configuration_string. Raises when it cannot be read: with fp07_response
    configured, an unidentified file must not match a correction by time alone.
    """
    from odas_tpw.chi.fp07_pair import config_from_source, pair_identity
    from odas_tpw.rsi.p_file import PFile

    if isinstance(source, PFile):
        return pair_identity(source.config, therm_names)
    return pair_identity(config_from_source(source), therm_names)


def _select_pair_correction(
    fp07_response: Sequence[Any],
    fp07_tau_scale: Mapping[str, float] | None,
    data: dict[str, Any],
    source: Any,
    therm_names: list[str],
) -> Any:
    """The fp07_response correction for this file, or raise."""
    from odas_tpw.chi import fp07_pair

    if fp07_tau_scale:
        raise ValueError(
            "fp07_response and fp07_tau_scale are mutually exclusive: the correction "
            "is measured relative to the bead pair, so applying it on top of a tau "
            "scale corrects the relative part twice"
        )
    corrections = fp07_pair.load_corrections(list(fp07_response))
    try:
        instrument, serials = _pair_identity(source, therm_names)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"fp07_response is configured but this file's instrument and beads cannot be "
            f"identified: {exc}"
        ) from exc
    start = str(data["metadata"].get("start_time", ""))
    if not start:
        raise ValueError("fp07_response needs the file's start_time to match a correction")
    t0 = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
    corr, why = fp07_pair.select(corrections, instrument, therm_names, serials, t0)
    if corr is None:
        raise ValueError(
            f"fp07_response is configured but no correction matches {instrument} channels "
            f"{therm_names} (beads {serials}) at {start}: " + "; ".join(why)
        )
    return corr


def _pair_provenance(
    corr: Any, applied: bool, delta: float, therm_names: list[str], reason: str = ""
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attributes and small variables recording the correction in the product.

    The factor itself is stored (f_ref, ln_e per probe, W_ref, f_hi), so a chi
    file reproduces its response without the correction file.
    """
    attrs: dict[str, Any] = {
        "fp07_response_applied": int(applied),
        "fp07_response_digest": corr.digest(),
        "fp07_response_instrument": corr.instrument,
        "fp07_response_serials": "/".join(corr.serials),
        "fp07_response_channels": "/".join(corr.channels),
        "fp07_response_span": f"{fp07_pair_iso(corr.t_start)}..{fp07_pair_iso(corr.t_end)}",
        "fp07_response_W_ref": corr.W_ref,
        "fp07_response_band": f"{corr.f_lo:.3f}-{corr.f_hi:.3f} Hz",
        "fp07_response_anchor": corr.anchor,
        "fp07_response_tau_model": corr.tau_model,
        "fp07_response_guard_delta": float(delta),
    }
    if not applied:
        attrs["fp07_response_reason"] = reason
        return attrs, {}
    ln_e = np.zeros((len(therm_names), corr.f_ref.size))
    for k, ch in enumerate(corr.channels):
        ln_e[therm_names.index(ch)] = corr.ln_e[k]
    variables = {
        "fp07_response_f": (
            ("fp07_response_freq",),
            np.asarray(corr.f_ref),
            {"units": "Hz", "long_name": "FP07 response correction frequency at W_ref"},
        ),
        "fp07_response_ln_e": (
            ("probe", "fp07_response_freq"),
            ln_e,
            {"long_name": "ln of each probe's response relative to the FP07 model",
             "comment": "|H_i|^2 = |H_model|^2 * e_i; chi divided each probe's observed and "
                        "noise spectra by e_i, with ln e_i weighted by the bin's S/N "
                        "(fp07_pair.signal_weighted; spec_gradT/spec_noise here are raw). "
                        "Applied at speed W by f -> f*tau(W)/tau(W_ref); held outside the band"},
        ),
    }
    return attrs, variables


def fp07_pair_iso(t: float) -> str:
    return datetime.fromtimestamp(float(t), UTC).isoformat(timespec="seconds")


def _select_slow_response(
    spans: Mapping[str, Any], data: dict[str, Any], source: Any, therm_names: list[str]
) -> Any:
    """Which configured slow paths this file's beads carry at its start time.

    The beads are read exactly as ``fp07_pair`` reads them, so a bead key and
    a pair correction cannot name different channels. An unreadable
    configuration is an error, never "no bead here": the bead may well be on it.
    """
    from odas_tpw.chi import fp07_slow

    try:
        _instrument, serials = _pair_identity(source, therm_names)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"fp07_slow_response is configured but this file's beads cannot be "
            f"identified: {exc}"
        ) from exc
    start = str(data["metadata"].get("start_time", ""))
    if not start:
        raise ValueError("fp07_slow_response needs the file's start_time to match a span")
    t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=UTC)  # the .p clock is UTC; never the local zone
    return fp07_slow.select(spans, serials, t0.timestamp())


def _apply_slow_response(
    sel: Any,
    spans: Mapping[str, Any],
    sections: Sequence[tuple[Any, Any]],
    therm_names: list[str],
    fname: str,
) -> dict[str, Any]:
    """Guard, attach the factors to every section's L3, and return the attributes.

    The guard's measured change is recorded for every configured bead found on
    the file, corrected or not (a bead outside its spans is still worth a QC
    number); it warns only where the correction is applied. A file that carries
    none of the configured beads gets only ``fp07_slow_response_serials``, the
    beads it does carry, so a run summary can say what a bead key that matched
    nothing was compared against; its chi is untouched.
    """
    from odas_tpw.chi import fp07_slow as S

    serials = {f"{S.ATTR}_serials": "/".join(sel.serials)}
    if not sel.applied and not sel.outside:
        return serials
    n = len(therm_names)
    present = {**{ci: s.bead for ci, s in sel.applied.items()}, **sel.outside}
    attrs: dict[str, Any] = {
        **serials,
        f"{S.ATTR}_applied": int(bool(sel.applied)),
        f"{S.ATTR}_model": S.MODEL_TEXT,
        f"{S.ATTR}_guard_method": S.GUARD_TEXT,
        f"{S.ATTR}_guard_limit": float(S.SLOW_GUARD_MAX),
        f"{S.ATTR}_guard_min_windows": int(S.SLOW_GUARD_MIN_WINDOWS),
    }
    measured: dict[int, float] = {}
    n_windows: dict[int, int] = {}
    skipped: dict[int, str] = {}
    for ci in present:
        ch = therm_names[ci]
        measured[ci], n_windows[ci] = float("nan"), 0
        if n != 2:
            skipped[ci] = f"skipped: the guard needs exactly two thermistors, this file has {n}"
            continue
        los, his = [], []
        for l2c, _l3c in sections:
            lo, hi = S.window_log_ratios(
                l2c.gradt[ci], l2c.gradt[1 - ci], l2c.section_number, l2c.fs_fast
            )
            los.append(lo)
            his.append(hi)
        lo_all = np.concatenate(los) if los else np.zeros(0)
        hi_all = np.concatenate(his) if his else np.zeros(0)
        measured[ci], n_windows[ci] = S.guard_delta(lo_all, hi_all), int(lo_all.size)
        attrs[f"{S.ATTR}_guard_windows_{ch}"] = n_windows[ci]
        if lo_all.size:
            # Each window's ln ratios, so a caller holding several profiles of
            # one file (perturb) can pool them into a file-level verdict.
            attrs[f"{S.ATTR}_guard_lnr_lo_{ch}"] = lo_all
            attrs[f"{S.ATTR}_guard_lnr_hi_{ch}"] = hi_all

    for ci, span in sorted(sel.applied.items()):
        ch = therm_names[ci]
        partner = sel.applied.get(1 - ci) if n == 2 else None
        predicted = span.ln_change() - (partner.ln_change() if partner is not None else 0.0)
        attrs.update({
            f"{S.ATTR}_bead_{ch}": span.bead,
            f"{S.ATTR}_beta_{ch}": float(span.beta),
            f"{S.ATTR}_tau_s_ms_{ch}": float(span.tau_s_ms),
            f"{S.ATTR}_span_{ch}": span.span_text(),
            f"{S.ATTR}_delta_measured_{ch}": float(measured[ci]),
            f"{S.ATTR}_delta_predicted_{ch}": float(predicted),
        })
        note = skipped.get(ci) or S.guard_verdict(measured[ci], predicted, n_windows[ci])
        if note.startswith("WARNING"):
            logger.warning("%s: fp07_slow_response %s on %s: %s", fname, span.bead, ch, note)
        attrs[f"{S.ATTR}_guard_{ch}"] = note
    reasons = []
    for ci, bead in sorted(sel.outside.items()):
        ch = therm_names[ci]
        attrs[f"{S.ATTR}_bead_{ch}"] = bead
        attrs[f"{S.ATTR}_delta_measured_{ch}"] = float(measured[ci])
        attrs[f"{S.ATTR}_guard_{ch}"] = skipped.get(ci, "not applied: recorded for QC only")
        listed = ", ".join(s.span_text() for s in spans.get(bead, ()))
        reasons.append(
            f"bead {bead} on {ch}: the file's start {fp07_pair_iso(sel.t)} is outside every "
            f"configured span ({listed})"
        )
    if reasons:
        attrs[f"{S.ATTR}_reason"] = "; ".join(reasons)
        logger.info("%s: fp07_slow_response not applied to %s", fname, "; ".join(reasons))
    if sel.applied:
        for _l2c, l3c in sections:
            S.apply_to_l3(l3c, sel, n)
    return attrs


def _load_therm_channels(
    source: PFile | str | Path,
    temperature_name: str | float = "auto",
    conductivity_name: str = "auto",
) -> dict[str, Any]:
    """Load thermistor gradient channels from any source.

    Looks for channels matching T*_dT* pattern (pre-emphasized temperature),
    or falls back to T1/T2 channels for first-difference gradient.
    ``temperature_name``/``conductivity_name`` select the slow reference
    temperature / conductivity channels (see :func:`load_channels`); they do
    not affect the fast thermistor-gradient channels gathered here.

    ``diff_gains``/``therm_cal`` come from the .p config for PFile sources
    and from the per-profile variable attrs written by ``extract_profiles``
    for NetCDF sources (#131 m8); a NetCDF file without those attrs (written
    before their introduction) falls back to the generic defaults with a
    logged warning. A PFile whose beads perturb fitted in situ gives each
    fitted bead's coefficients instead of the embedded ones (#245), and
    ``therm_cal_source`` names them per channel (None: embedded).
    """
    from odas_tpw.rsi.helpers import load_channels
    from odas_tpw.rsi.p_file import PFile

    # First load standard channels (convert TypedDict → plain dict for extra keys)
    data: dict[str, Any] = dict(
        load_channels(
            source, temperature_name=temperature_name, conductivity_name=conductivity_name
        )
    )

    therm: list[tuple[str, np.ndarray]] = []
    diff_gains: list[float] = []

    if isinstance(source, PFile):
        pf = source
    elif Path(source).suffix.lower() == ".p":
        pf = PFile(source)
    else:
        pf = None

    therm_cal: list[dict[str, float]] = []
    therm_cal_source: list[str | None] = []

    if pf is not None:
        from odas_tpw.rsi.helpers import DT_PATTERN, T_PATTERN

        for name in sorted(pf._fast_channels):
            if DT_PATTERN.match(name):
                therm.append((name, pf.channels[name]))
                # diff_gain from the gradient channel's config; thermistor
                # calibration from the base channel (T1 for T1_dT1). Shared
                # with the per-profile NetCDF write path (#131 m8 F7).
                dg, cal = _therm_gradient_config(pf.config, name, pf.channel_info)
                diff_gains.append(dg)
                therm_cal.append(cal)
                therm_cal_source.append(_bead_coeffs_source(pf.channel_info, name))

        # If no pre-emphasized channels, use T channels with first-difference
        if not therm:
            for name in sorted(pf._fast_channels):
                if T_PATTERN.match(name):
                    # Compute gradient via first-difference
                    T_data = pf.channels[name]
                    therm.append((name, T_data))
                    diff_gains.append(_DEFAULT_DIFF_GAIN)
                    ch_cfg: dict = next(
                        (ch for ch in pf.config["channels"] if ch.get("name") == name),
                        {},
                    )
                    therm_cal.append(_extract_therm_cal(ch_cfg, _bead_fit(pf.channel_info, name)))
                    therm_cal_source.append(_bead_coeffs_source(pf.channel_info, name))
    else:
        # NetCDF source — look for gradient channels or T channels.
        # _nc_filled: masked _FillValue samples become NaN, never the raw
        # ~9.97e36 fill buffer (see _channels_from_nc).
        import netCDF4 as nc

        from odas_tpw.rsi.helpers import DT_PATTERN, T_PATTERN
        from odas_tpw.rsi.profile import _nc_filled

        assert not isinstance(source, PFile)  # pf is None only for path sources
        ds = nc.Dataset(str(source), "r")

        # Per-profile files written by extract_profiles carry the FP07
        # coefficients as variable attrs (#131 m8): ``diff_gain`` plus the
        # THERM_CAL_ATTR_KEYS subset. Two kinds: e_b, b, gain and adc_* describe
        # the ELECTRONICS and are always the embedded config's; T_0, beta_1 and
        # beta_2 are the BEAD's R(T), which converts the noise counts to kelvin,
        # and are the coefficients the channel's data were converted with --
        # perturb's in-situ fit where it fitted one (BEAD_COEFFS_SOURCE_ATTR
        # names it, beta_3 joins them for an order-3 fit, and the noise model
        # inverts that curve exactly), the embedded config's otherwise (#245).
        missing_attr_vars: list[str] = []
        stale_fit_vars: list[str] = []

        def _read_therm_attrs(vname: str, var: Any) -> None:
            attr_names = set(var.ncattrs())
            if "diff_gain" in attr_names:
                diff_gains.append(float(var.getncattr("diff_gain")))
            else:
                # Older per-profile file (pre-m8) — fall back, never silently.
                diff_gains.append(_DEFAULT_DIFF_GAIN)
                missing_attr_vars.append(vname)
            therm_cal.append(
                {k: float(var.getncattr(k)) for k in THERM_CAL_ATTR_KEYS if k in attr_names}
            )
            if BEAD_COEFFS_SOURCE_ATTR in attr_names:
                therm_cal_source.append(str(var.getncattr(BEAD_COEFFS_SOURCE_ATTR)))
            else:
                therm_cal_source.append(None)
                # Data fitted in situ, coefficients embedded: written before #245.
                if "diff_gain" in attr_names and str(
                    getattr(var, "calibration", "")
                ).startswith("in-situ"):
                    stale_fit_vars.append(vname)

        for vname in sorted(ds.variables.keys()):
            var = ds.variables[vname]
            # Union of PR #137 (masked samples -> NaN via _nc_filled) and
            # PR #138 (embedded FP07 calibration attrs).
            if var.dimensions == ("time_fast",) and DT_PATTERN.match(vname):
                therm.append((vname, _nc_filled(var)))
                _read_therm_attrs(vname, var)

        if not therm:
            for vname in sorted(ds.variables.keys()):
                var = ds.variables[vname]
                if var.dimensions == ("time_fast",) and T_PATTERN.match(vname):
                    therm.append((vname, _nc_filled(var)))
                    _read_therm_attrs(vname, var)

        ds.close()

        if missing_attr_vars:
            logger.warning(
                "%s: no FP07 calibration attributes on %s (per-profile file "
                "predates their introduction); falling back to generic "
                "diff_gain=%.2f and default thermistor coefficients — "
                "re-extract profiles from the .p file to embed the factory "
                "values",
                Path(source).name,
                ", ".join(missing_attr_vars),
                _DEFAULT_DIFF_GAIN,
            )
        if stale_fit_vars:
            logger.warning(
                "%s: %s carry in-situ calibrated data but the EMBEDDED bead "
                "coefficients (per-profile file written before issue #245), so "
                "the chi noise model is on a different calibration from the "
                "data; re-extract the profiles to fix it",
                Path(source).name,
                ", ".join(stale_fit_vars),
            )

    data["therm"] = therm
    data["diff_gains"] = diff_gains
    data["therm_cal"] = therm_cal if therm_cal else [{}] * len(therm)
    data["therm_cal_source"] = therm_cal_source
    return data


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------


def compute_chi_file(
    source_path: str | Path,
    output_dir: str | Path,
    **chi_kwargs: Any,
) -> list[Path]:
    """Compute chi for one file and write NetCDF output(s).

    Parameters
    ----------
    source_path : str or Path
        Input file.
    output_dir : str or Path
        Output directory.
    **chi_kwargs
        Keyword arguments forwarded to ``_compute_chi`` (the chi parameter
        set; ``get_chi`` is the deprecated public wrapper).

    Returns
    -------
    list of Path
        Paths to output files written.
    """
    from odas_tpw.rsi.helpers import write_profile_results

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    results = _compute_chi(source_path, **chi_kwargs)
    return write_profile_results(results, source_path, output_dir, "chi")


def load_epsilon_dataset(
    source_path: str | Path, epsilon_dir: str | Path
) -> xr.Dataset | None:
    """Locate and open the epsilon dataset(s) for *source_path* (Method 1).

    ``rsi-tpw eps`` writes into hash-tracked subdirectories
    (``epsilon_dir/eps_00/``...) and names files per profile
    (``{stem}_prof001_eps.nc``).  This helper searches *epsilon_dir*
    itself first, then its ``eps_*`` subdirectories (most recently
    modified first), and concatenates multiple per-profile files along
    ``time`` so every chi window can pair with its own profile's epsilon
    estimates.

    Returns None when no matching epsilon files exist (the caller falls
    back to Method 2).
    """
    stem = Path(source_path).stem
    base = Path(epsilon_dir)
    candidates = [
        base,
        *sorted(
            (d for d in base.glob("eps_*") if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        ),
    ]
    for d in candidates:
        files = sorted(d.glob(f"{stem}_prof*_eps.nc"))
        # The single-profile (`{stem}_eps.nc`) and per-profile
        # (`{stem}_prof00N_eps.nc`) naming schemes are mutually exclusive within
        # one run. Only fall back to the single-profile file when NO per-profile
        # file exists: concatenating both (e.g. when a re-run with different
        # trim/direction left stale products of both forms in the dir) would
        # double-count epsilon times and silently bias the Method-1 match.
        if not files:
            single = d / f"{stem}_eps.nc"
            if single.exists():
                files = [single]
        if not files:
            continue
        if len(files) == 1:
            return xr.open_dataset(files[0])
        datasets = [xr.open_dataset(p).load() for p in files]
        try:
            return xr.concat(datasets, dim="time", combine_attrs="override")
        finally:
            for ds in datasets:
                ds.close()
    return None


def slow_response_summary(paths: Sequence[str | Path]) -> dict[str, Any] | None:
    """The ``fp07_slow_response`` outcome of one chi call, from its first product.

    Every product of one ``_compute_chi`` call carries the same file-level
    attributes, so the first is enough. None for no products, or products made
    without the correction configured.
    """
    if not paths:
        return None
    import netCDF4

    from odas_tpw.chi.fp07_slow import ATTR, product_summary

    with netCDF4.Dataset(str(paths[0])) as ds:
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs() if k.startswith(ATTR)}
    return product_summary(attrs)


def _compute_chi_one(args: tuple) -> tuple[str, int, dict[str, Any] | None]:
    """Worker for parallel chi computation.

    args is (source_path, output_dir, kwargs) or
           (source_path, output_dir, kwargs, epsilon_dir).
    When epsilon_dir is provided, the worker opens the matching epsilon
    file itself (xr.Datasets cannot be pickled across process boundaries).
    Returns (source, number of products, :func:`slow_response_summary` when
    ``fp07_slow_response`` is configured, else None).
    """
    if len(args) == 4:
        source_path, output_dir, kwargs, epsilon_dir = args
    else:
        source_path, output_dir, kwargs = args
        epsilon_dir = None

    eps_ds = load_epsilon_dataset(source_path, epsilon_dir) if epsilon_dir else None
    if eps_ds is not None:
        try:
            kwargs = dict(kwargs)
            kwargs["epsilon_ds"] = eps_ds
            paths = compute_chi_file(source_path, output_dir, **kwargs)
        finally:
            eps_ds.close()
    else:
        if epsilon_dir is not None:
            print(
                f"  Warning: no epsilon files for {Path(source_path).stem} "
                f"under {epsilon_dir} (searched eps_* subdirectories too); "
                "using Method 2"
            )
        paths = compute_chi_file(source_path, output_dir, **kwargs)
    slow = slow_response_summary(paths) if kwargs.get("fp07_slow_response") else None
    return str(source_path), len(paths), slow
