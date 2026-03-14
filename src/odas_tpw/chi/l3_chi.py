"""L3 chi: temperature gradient spectra for chi estimation.

Computes Welch-method spectra of temperature gradient, applies Goodman
coherent-noise removal, first-difference and bilinear corrections, and
converts to wavenumber units.

Reads cleaned temperature gradient and HP-filtered vibration from L2ChiData.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from odas_tpw.chi.chi import _bilinear_correction, _default_tau_model
from odas_tpw.chi.fp07 import gradT_noise_batch
from odas_tpw.chi.l2_chi import L2ChiData
from odas_tpw.scor160.goodman import clean_shear_spec_batch
from odas_tpw.scor160.io import L3Params
from odas_tpw.scor160.ocean import visc, visc35
from odas_tpw.scor160.spectral import csd_matrix_batch


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
        return self.time.shape[0]

    @property
    def n_wavenumber(self) -> int:
        return self.kcyc.shape[0]

    @property
    def n_gradt(self) -> int:
        return self.gradt_spec.shape[0]


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

    # Use pre-computed gradient from L2ChiData
    gradt_arrays = l2_chi.gradt

    # Frequency vector (constant)
    F_const = np.arange(n_freq) * fs / nfft

    # First-difference correction (constant across windows/probes)
    fd_correction = np.ones(n_freq)
    with np.errstate(divide="ignore", invalid="ignore"):
        fd_correction[1:] = (np.pi * F_const[1:] / (fs * np.sin(np.pi * F_const[1:] / fs))) ** 2
    fd_correction = np.where(np.isfinite(fd_correction), fd_correction, 1.0)

    # Bilinear corrections per probe
    bl_corrections = [_bilinear_correction(F_const, dg, fs) for dg in diff_gains]

    # Collect windows from all sections
    sections = np.unique(l2_chi.section_number)
    sections = sections[sections > 0]

    all_times = []
    all_pres = []
    all_temp = []
    all_speed = []
    all_section = []
    all_gradt_spec = []
    all_noise_spec = []
    all_kcyc = []
    all_H2 = []
    all_tau0 = []
    all_nu = []

    # HP-filtered vib from L2ChiData for Goodman cleaning
    vib = l2_chi.vib
    n_vib = l2_chi.n_vib
    do_goodman = params.goodman and n_vib > 0

    for sec_id in sections:
        mask = l2_chi.section_number == sec_id
        idx = np.where(mask)[0]
        if len(idx) < diss_length:
            continue

        sec_start = idx[0]
        sec_end = idx[-1] + 1
        sec_len = sec_end - sec_start

        n_windows = (sec_len - diss_length) // diss_step + 1
        if n_windows < 1:
            continue

        starts = sec_start + np.arange(n_windows) * diss_step

        # Build gradient windows: (n_windows, diss_length, n_temp)
        indices = starts[:, np.newaxis] + np.arange(diss_length)[np.newaxis, :]
        gradt_windows = np.stack(
            [gradt_arrays[ci][indices] for ci in range(n_temp)],
            axis=-1,
        )

        # Goodman cleaning with HP-filtered vibration from L2ChiData
        if do_goodman:
            vib_windows = np.stack(
                [vib[vi][indices] for vi in range(n_vib)],
                axis=-1,
            )
            clean_TT, _ = clean_shear_spec_batch(
                vib_windows,
                gradt_windows,
                nfft,
                fs,
            )
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

        # Wavenumber: K = F / W
        K_all = F_const[:, np.newaxis] / speed_means[np.newaxis, :]

        # Apply corrections to spectra
        for ci in range(n_temp):
            # clean_spectra: (n_windows, n_freq, n_temp)
            clean_spectra[:, :, ci] *= (
                speed_means[:, np.newaxis]
                * fd_correction[np.newaxis, :]
                * bl_corrections[ci][np.newaxis, :]
            )

        # FP07 transfer function
        tau_model = _default_tau_model(fp07_model)
        if tau_model == "lueck":
            tau0_all = 0.01 * (1.0 / speed_means) ** 0.5
        elif tau_model == "peterson":
            tau0_all = 0.012 * speed_means ** (-0.32)
        else:
            tau0_all = np.full(n_windows, 0.003)

        omega_tau = (2 * np.pi * F_const[:, np.newaxis] * tau0_all[np.newaxis, :]) ** 2
        if fp07_model == "single_pole":
            H2_all = (1.0 / (1.0 + omega_tau)).T
        else:
            H2_all = (1.0 / (1.0 + omega_tau) ** 2).T

        # Noise spectra
        noise_all = np.empty((n_temp, n_windows, n_freq))
        for ci in range(n_temp):
            cal_ci = therm_cal[ci] if therm_cal and ci < len(therm_cal) else {}
            noise_kwargs = {
                k: v
                for k, v in cal_ci.items()
                if k in ("e_b", "gain", "beta_1", "adc_fs", "adc_bits")
            }
            noise_all[ci] = gradT_noise_batch(
                F_const,
                T_means,
                speed_means,
                fs=fs,
                diff_gain=diff_gains[ci],
                **noise_kwargs,
            )

        # Store results
        for w in range(n_windows):
            all_times.append(t_means[w])
            all_pres.append(P_means[w])
            all_temp.append(T_means[w])
            all_speed.append(speed_means[w])
            all_section.append(sec_id)
            all_nu.append(nu_all[w])

            all_kcyc.append(K_all[:, w])
            all_H2.append(H2_all[w])
            all_tau0.append(tau0_all[w])

            # gradt_spec: (n_temp, n_freq)
            spec_w = clean_spectra[w].T  # (n_temp, n_freq)
            all_gradt_spec.append(spec_w)

            noise_w = noise_all[:, w, :]  # (n_temp, n_freq)
            all_noise_spec.append(noise_w)

    if not all_times:
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
    time_out = np.array(all_times)
    pres_out = np.array(all_pres)
    temp_out = np.array(all_temp)
    speed_out = np.array(all_speed)
    section_out = np.array(all_section)
    nu_out = np.array(all_nu)
    tau0_out = np.array(all_tau0)

    kcyc_out = np.column_stack(all_kcyc)  # (n_freq, n_spec)
    H2_out = np.stack(all_H2, axis=0)  # (n_spec, n_freq)

    gradt_spec_out = np.stack(all_gradt_spec, axis=-1)  # (n_temp, n_freq, n_spec)
    noise_spec_out = np.stack(all_noise_spec, axis=-1)  # (n_temp, n_freq, n_spec)

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
