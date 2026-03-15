"""L3 chi: temperature gradient spectra for chi estimation.

Computes Welch-method spectra of temperature gradient, applies Goodman
coherent-noise removal, first-difference and bilinear corrections, and
converts to wavenumber units.

Reads cleaned temperature gradient and HP-filtered vibration from L2ChiData.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from odas_tpw.chi.chi import _bilinear_correction
from odas_tpw.chi.fp07 import (
    default_tau_model,
    fp07_tau_batch,
    fp07_transfer_batch,
    gradT_noise_batch,
)
from odas_tpw.chi.l2_chi import L2ChiData
from odas_tpw.scor160.goodman import clean_shear_spec_batch
from odas_tpw.scor160.io import L3Params
from odas_tpw.scor160.ocean import visc, visc35
from odas_tpw.scor160.spectral import csd_matrix_batch


@dataclass
class _SectionResult:
    """Accumulator for per-window results within a section."""

    times: list = field(default_factory=list)
    pres: list = field(default_factory=list)
    temp: list = field(default_factory=list)
    speed: list = field(default_factory=list)
    section: list = field(default_factory=list)
    gradt_spec: list = field(default_factory=list)
    noise_spec: list = field(default_factory=list)
    kcyc: list = field(default_factory=list)
    H2: list = field(default_factory=list)
    tau0: list = field(default_factory=list)
    nu: list = field(default_factory=list)


@dataclass
class L3ChiData:
    """Level-3 temperature gradient spectra for chi estimation."""

    time: np.ndarray  # (N_SPECTRA,), center time per window
    pres: np.ndarray  # (N_SPECTRA,), mean pressure per window
    temp: np.ndarray  # (N_SPECTRA,), mean temperature per window
    pspd_rel: np.ndarray  # (N_SPECTRA,), mean speed per window
    section_number: np.ndarray  # (N_SPECTRA,)
    nu: np.ndarray  # (N_SPECTRA,), kinematic viscosity

    kcyc: np.ndarray  # (N_WAVENUMBER, N_SPECTRA)
    freq: np.ndarray  # (N_FREQ,), frequency vector [Hz]
    gradt_spec: np.ndarray  # (N_GRADT, N_WAVENUMBER, N_SPECTRA)
    noise_spec: np.ndarray  # (N_GRADT, N_WAVENUMBER, N_SPECTRA)
    H2: np.ndarray  # (N_SPECTRA, N_FREQ), FP07 transfer function
    tau0: np.ndarray  # (N_SPECTRA,), FP07 time constant

    diff_gains: list[float] = field(default_factory=list)
    fp07_model: str = "single_pole"

    @property
    def n_spectra(self) -> int:
        """Number of spectral windows."""
        return int(self.time.shape[0])

    @property
    def n_wavenumber(self) -> int:
        """Number of wavenumber bins."""
        return int(self.kcyc.shape[0])

    @property
    def n_gradt(self) -> int:
        """Number of temperature gradient channels."""
        return int(self.gradt_spec.shape[0])


def _process_section_chi(
    l2_chi: L2ChiData,
    sec_id: int,
    diss_length: int,
    diss_step: int,
    nfft: int,
    fs: float,
    n_temp: int,
    n_freq: int,
    do_goodman: bool,
    fd_correction: np.ndarray,
    bl_corrections: list[np.ndarray],
    F_const: np.ndarray,
    fp07_model: str,
    salinity: float | np.ndarray | None,
    diff_gains: list[float],
    therm_cal: list[dict] | None,
    acc: _SectionResult,
) -> None:
    """Process a single section, appending results to *acc*."""
    mask = l2_chi.section_number == sec_id
    idx = np.where(mask)[0]
    if len(idx) < diss_length:
        return

    sec_start = idx[0]
    sec_end = idx[-1] + 1
    sec_len = sec_end - sec_start

    n_windows = (sec_len - diss_length) // diss_step + 1
    if n_windows < 1:
        return

    starts = sec_start + np.arange(n_windows) * diss_step
    indices = starts[:, np.newaxis] + np.arange(diss_length)[np.newaxis, :]

    gradt_windows = np.stack(
        [l2_chi.gradt[ci][indices] for ci in range(n_temp)],
        axis=-1,
    )

    # Goodman cleaning
    n_vib = l2_chi.n_vib
    if do_goodman:
        vib_windows = np.stack(
            [l2_chi.vib[vi][indices] for vi in range(n_vib)],
            axis=-1,
        )
        clean_TT, _ = clean_shear_spec_batch(vib_windows, gradt_windows, nfft, fs)
        clean_spectra = np.real(np.diagonal(clean_TT, axis1=2, axis2=3)).copy()
    else:
        TT, _, _, _ = csd_matrix_batch(
            gradt_windows,
            None,
            nfft,
            fs,
            overlap=nfft // 2,
            detrend="linear",
        )
        clean_spectra = np.real(np.diagonal(TT, axis1=2, axis2=3)).copy()

    # Per-window means
    speed_means = np.maximum(np.mean(np.abs(l2_chi.pspd_rel[indices]), axis=1), 0.01)
    P_means = np.mean(l2_chi.pres[indices], axis=1)
    t_means = np.mean(l2_chi.time[indices], axis=1)
    T_means = np.mean(l2_chi.temp[indices], axis=1)

    # Viscosity
    if salinity is not None:
        if np.ndim(salinity) > 0:
            sal_means = np.mean(np.asarray(salinity)[indices], axis=1)
        else:
            sal_means = np.full(n_windows, float(salinity))
        nu_all = np.asarray(
            [visc(T_means[i], sal_means[i], P_means[i]) for i in range(n_windows)],
            dtype=np.float64,
        )
    else:
        nu_all = np.asarray(
            [visc35(T_means[i]) for i in range(n_windows)],
            dtype=np.float64,
        )

    K_all = F_const[:, np.newaxis] / speed_means[np.newaxis, :]

    # Apply corrections
    for ci in range(n_temp):
        clean_spectra[:, :, ci] *= (
            speed_means[:, np.newaxis]
            * fd_correction[np.newaxis, :]
            * bl_corrections[ci][np.newaxis, :]
        )

    # FP07 transfer function
    tau_model = default_tau_model(fp07_model)
    tau0_all = fp07_tau_batch(speed_means, model=tau_model)
    H2_all = fp07_transfer_batch(F_const, tau0_all, model=fp07_model)

    # Noise spectra
    noise_all = np.empty((n_temp, n_windows, n_freq))
    for ci in range(n_temp):
        cal_ci = therm_cal[ci] if therm_cal and ci < len(therm_cal) else {}
        noise_kwargs = {
            k: v for k, v in cal_ci.items() if k in ("e_b", "gain", "beta_1", "adc_fs", "adc_bits")
        }
        noise_all[ci] = gradT_noise_batch(
            F_const,
            T_means,
            speed_means,
            fs=fs,
            diff_gain=diff_gains[ci],
            **noise_kwargs,
        )

    # Append results
    for w in range(n_windows):
        acc.times.append(t_means[w])
        acc.pres.append(P_means[w])
        acc.temp.append(T_means[w])
        acc.speed.append(speed_means[w])
        acc.section.append(sec_id)
        acc.nu.append(nu_all[w])
        acc.kcyc.append(K_all[:, w])
        acc.H2.append(H2_all[w])
        acc.tau0.append(tau0_all[w])
        acc.gradt_spec.append(clean_spectra[w].T)
        acc.noise_spec.append(noise_all[:, w, :])


def process_l3_chi(
    l2_chi: L2ChiData,
    params: L3Params,
    *,
    fp07_model: str = "single_pole",
    salinity: float | np.ndarray | None = None,
    therm_cal: list[dict] | None = None,
) -> L3ChiData:
    """Compute L3 temperature gradient spectra from L2ChiData.

    Parameters
    ----------
    l2_chi : L2ChiData
        Level-2 cleaned chi data (despiked temperature gradient,
        HP-filtered vibration, section info).
    params : L3Params
        Spectral parameters (fft_length, diss_length, overlap).
    fp07_model : str
        FP07 transfer function model ('single_pole' or 'double_pole').
    salinity : float or array or None
        Practical salinity for viscosity.
    therm_cal : list of dict, optional
        Per-thermistor calibration for noise model.

    Returns
    -------
    L3ChiData
    """
    fs = params.fs_fast
    nfft = params.fft_length
    diss_length = params.diss_length
    diss_overlap = params.overlap
    diss_step = diss_length - diss_overlap
    n_freq = nfft // 2 + 1
    n_temp = l2_chi.n_temp
    diff_gains = l2_chi.diff_gains

    # Frequency vector (constant)
    F_const = np.arange(n_freq) * fs / nfft

    # First-difference correction (constant across windows/probes)
    fd_correction = np.ones(n_freq)
    with np.errstate(divide="ignore", invalid="ignore"):
        fd_correction[1:] = (np.pi * F_const[1:] / (fs * np.sin(np.pi * F_const[1:] / fs))) ** 2
    fd_correction = np.where(np.isfinite(fd_correction), fd_correction, 1.0)

    # Bilinear corrections per probe
    bl_corrections = [_bilinear_correction(F_const, dg, fs) for dg in diff_gains]

    do_goodman = params.goodman and l2_chi.n_vib > 0
    sections = np.unique(l2_chi.section_number)
    sections = sections[sections > 0]

    acc = _SectionResult()

    for sec_id in sections:
        _process_section_chi(
            l2_chi,
            sec_id,
            diss_length,
            diss_step,
            nfft,
            fs,
            n_temp,
            n_freq,
            do_goodman,
            fd_correction,
            bl_corrections,
            F_const,
            fp07_model,
            salinity,
            diff_gains,
            therm_cal,
            acc,
        )

    if not acc.times:
        return L3ChiData(
            time=np.array([]),
            pres=np.array([]),
            temp=np.array([]),
            pspd_rel=np.array([]),
            section_number=np.array([]),
            nu=np.array([]),
            kcyc=np.zeros((n_freq, 0)),
            freq=F_const,
            gradt_spec=np.zeros((n_temp, n_freq, 0)),
            noise_spec=np.zeros((n_temp, n_freq, 0)),
            H2=np.zeros((0, n_freq)),
            tau0=np.array([]),
            diff_gains=diff_gains,
            fp07_model=fp07_model,
        )

    # Assemble output arrays
    time_out = np.array(acc.times)
    pres_out = np.array(acc.pres)
    temp_out = np.array(acc.temp)
    speed_out = np.array(acc.speed)
    section_out = np.array(acc.section)
    nu_out = np.array(acc.nu)
    tau0_out = np.array(acc.tau0)

    kcyc_out = np.column_stack(acc.kcyc)  # (n_freq, n_spec)
    H2_out = np.stack(acc.H2, axis=0)  # (n_spec, n_freq)

    gradt_spec_out = np.stack(acc.gradt_spec, axis=-1)  # (n_temp, n_freq, n_spec)
    noise_spec_out = np.stack(acc.noise_spec, axis=-1)  # (n_temp, n_freq, n_spec)

    return L3ChiData(
        time=time_out,
        pres=pres_out,
        temp=temp_out,
        pspd_rel=speed_out,
        section_number=section_out,
        nu=nu_out,
        kcyc=kcyc_out,
        freq=F_const,
        gradt_spec=gradt_spec_out,
        noise_spec=noise_spec_out,
        H2=H2_out,
        tau0=tau0_out,
        diff_gains=diff_gains,
        fp07_model=fp07_model,
    )
