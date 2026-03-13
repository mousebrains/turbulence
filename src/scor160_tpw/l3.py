"""L2 → L3: wavenumber spectra from cleaned time series.

Implements Sec. 3.3 of Lueck et al. (2024).

Steps:
  1. Divide each section into overlapping dissipation windows.
  2. Compute frequency spectra (Welch method, cosine window, 50% FFT overlap).
  3. Apply Goodman coherent-noise removal using vibration/accel signals.
  4. Convert frequency spectra to wavenumber spectra using mean segment speed.
  5. Record per-window pressure, temperature, speed, section number.
"""

from __future__ import annotations

import numpy as np

from rsi_python.goodman import clean_shear_spec_batch
from rsi_python.spectral import csd_matrix_batch
from scor160_tpw.io import L1Data, L2Data, L3Data, L3Params

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_l3(l2: L2Data, l1: L1Data, params: L3Params) -> L3Data:
    """Compute L3 wavenumber spectra from L2 cleaned time series.

    Parameters
    ----------
    l2 : L2Data
        Level-2 cleaned time series (despiked, HP-filtered).
    l1 : L1Data
        Level-1 data (needed for pressure, temperature, fs_fast).
    params : L3Params
        Spectral parameters (fft_length, diss_length, overlap, etc.).

    Returns
    -------
    L3Data with wavenumber spectra per dissipation window.
    """
    fs = params.fs_fast
    nfft = params.fft_length
    diss_length = params.diss_length
    diss_overlap = params.overlap
    diss_step = diss_length - diss_overlap
    n_freq = nfft // 2 + 1

    n_shear = l2.shear.shape[0]
    n_vib = l2.vib.shape[0]

    # Collect windows from all sections
    all_times = []
    all_pres = []
    all_temp = []
    all_speed = []
    all_section = []
    all_sh_spec = []
    all_sh_spec_clean = []
    all_kcyc = []

    sections = np.unique(l2.section_number)
    sections = sections[sections > 0]

    for sec_id in sections:
        mask = l2.section_number == sec_id
        idx = np.where(mask)[0]
        if len(idx) < diss_length:
            continue

        sec_start = idx[0]
        sec_end = idx[-1] + 1
        sec_len = sec_end - sec_start

        # Window start positions within the section
        n_windows = (sec_len - diss_length) // diss_step + 1
        if n_windows < 1:
            continue

        starts = sec_start + np.arange(n_windows) * diss_step

        # Build windowed arrays: (n_windows, diss_length, n_channels)
        # Shear: L2 data is (N_SHEAR, N_TIME) → need (n_windows, diss_length, n_shear)
        shear_windows = np.zeros((n_windows, diss_length, n_shear), dtype=np.float64)
        for w in range(n_windows):
            s = starts[w]
            shear_windows[w] = l2.shear[:, s : s + diss_length].T

        # Vibration/acceleration windows
        vib_windows = None
        if n_vib > 0 and params.goodman:
            vib_windows = np.zeros((n_windows, diss_length, n_vib), dtype=np.float64)
            for w in range(n_windows):
                s = starts[w]
                vib_windows[w] = l2.vib[:, s : s + diss_length].T

        # Compute raw frequency spectra (Welch method)
        # Returns: (n_windows, n_freq, n_shear, n_shear) — diagonal is auto-spectrum
        Cxy, F, _, _ = csd_matrix_batch(
            shear_windows, None, nfft, fs,
            overlap=nfft // 2, detrend="linear",
        )
        # Extract auto-spectra: diagonal elements
        # Cxy shape: (n_windows, n_freq, n_shear, n_shear)
        sh_freq_spec = np.real(
            np.diagonal(Cxy, axis1=2, axis2=3)
        )  # (n_windows, n_freq, n_shear)

        # Goodman cleaning
        if vib_windows is not None and n_vib > 0:
            clean_UU, _ = clean_shear_spec_batch(
                vib_windows, shear_windows, nfft, fs,
            )
            # clean_UU: (n_windows, n_freq, n_shear, n_shear)
            sh_freq_spec_clean = np.real(
                np.diagonal(clean_UU, axis1=2, axis2=3)
            )  # (n_windows, n_freq, n_shear)
        else:
            sh_freq_spec_clean = sh_freq_spec.copy()

        # Per-window mean values
        for w in range(n_windows):
            s = starts[w]
            e = s + diss_length
            center_idx = s + diss_length // 2

            all_times.append(l2.time[center_idx])
            all_pres.append(np.mean(l1.pres[s:e]))
            if l1.temp.size > 0 and l1.temp.shape[0] == l1.n_time:
                all_temp.append(np.mean(l1.temp[s:e]))
            else:
                all_temp.append(np.nan)
            all_speed.append(np.mean(l2.pspd_rel[s:e]))
            all_section.append(sec_id)

            # Convert frequency spectrum → wavenumber spectrum
            # k = f / W [cpm], Ψ(k) = Ψ(f) * W [variance/cpm]
            W = all_speed[-1]
            W = max(W, 0.05)  # minimum speed to avoid infinities
            kcyc_w = F / W  # (n_freq,)
            all_kcyc.append(kcyc_w)

            # Frequency → wavenumber: multiply by W
            # Ψ(k) dk = Ψ(f) df, and k = f/W, dk = df/W
            # So Ψ(k) = Ψ(f) * W
            sh_k = sh_freq_spec[w] * W  # (n_freq, n_shear)
            sh_k_clean = sh_freq_spec_clean[w] * W

            all_sh_spec.append(sh_k)
            all_sh_spec_clean.append(sh_k_clean)

    if not all_times:
        # No valid windows
        return L3Data(
            time=np.array([]),
            pres=np.array([]),
            temp=np.array([]),
            pspd_rel=np.array([]),
            section_number=np.array([]),
            kcyc=np.zeros((n_freq, 0)),
            sh_spec=np.zeros((n_shear, n_freq, 0)),
            sh_spec_clean=np.zeros((n_shear, n_freq, 0)),
        )

    # Assemble arrays in ATOMIX format: (N_WAVENUMBER, N_SPECTRA) etc.
    time_out = np.array(all_times)
    pres_out = np.array(all_pres)
    temp_out = np.array(all_temp)
    speed_out = np.array(all_speed)
    section_out = np.array(all_section)

    # kcyc: (N_WAVENUMBER, N_SPECTRA)
    kcyc_out = np.column_stack(all_kcyc)  # (n_freq, n_spec)

    # sh_spec: list of (n_freq, n_shear) → (N_SHEAR, N_WAVENUMBER, N_SPECTRA)
    sh_spec_arr = np.stack(all_sh_spec, axis=-1)  # (n_freq, n_shear, n_spec)
    sh_spec_out = np.transpose(sh_spec_arr, (1, 0, 2))  # (n_shear, n_freq, n_spec)

    sh_spec_clean_arr = np.stack(all_sh_spec_clean, axis=-1)
    sh_spec_clean_out = np.transpose(sh_spec_clean_arr, (1, 0, 2))

    return L3Data(
        time=time_out,
        pres=pres_out,
        temp=temp_out,
        pspd_rel=speed_out,
        section_number=section_out,
        kcyc=kcyc_out,
        sh_spec=sh_spec_out,
        sh_spec_clean=sh_spec_clean_out,
    )
