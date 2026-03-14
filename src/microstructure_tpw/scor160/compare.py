"""Comparison utilities for validating computed results against reference."""

from __future__ import annotations

import numpy as np

from microstructure_tpw.scor160.io import L2Data, L3Data, L4Data


def compare_l2(computed: L2Data, reference: L2Data) -> dict:
    """Compare computed L2 against reference L2.

    Returns a dict of comparison metrics.
    """
    results: dict = {}

    # Section agreement
    ref_sec = reference.section_number
    comp_sec = computed.section_number
    ref_in = ref_sec > 0
    comp_in = comp_sec > 0
    both_in = ref_in & comp_in

    results["section"] = {
        "ref_n_sections": int(ref_sec.max()),
        "comp_n_sections": int(comp_sec.max()),
        "ref_n_selected": int(ref_in.sum()),
        "comp_n_selected": int(comp_in.sum()),
        "overlap_frac": float(both_in.sum() / max(ref_in.sum(), 1)),
    }

    # Speed comparison (within reference sections)
    if both_in.any():
        w_ref = reference.pspd_rel[both_in]
        w_comp = computed.pspd_rel[both_in]
        finite = np.isfinite(w_ref) & np.isfinite(w_comp)
        if finite.any():
            diff = w_comp[finite] - w_ref[finite]
            results["speed"] = {
                "mean_diff": float(np.mean(diff)),
                "rms_diff": float(np.sqrt(np.mean(diff**2))),
                "max_abs_diff": float(np.max(np.abs(diff))),
                "ref_mean": float(np.mean(w_ref[finite])),
                "relative_rms": float(
                    np.sqrt(np.mean(diff**2)) / np.mean(np.abs(w_ref[finite]))
                ),
            }

    # Shear comparison (within reference sections)
    n_sh = min(computed.shear.shape[0], reference.shear.shape[0])
    shear_stats = []
    for i in range(n_sh):
        ref_sh = reference.shear[i, both_in]
        comp_sh = computed.shear[i, both_in]
        finite = np.isfinite(ref_sh) & np.isfinite(comp_sh)
        if finite.any():
            diff = comp_sh[finite] - ref_sh[finite]
            ref_rms = np.sqrt(np.mean(ref_sh[finite] ** 2))
            shear_stats.append(
                {
                    "probe": i + 1,
                    "rms_diff": float(np.sqrt(np.mean(diff**2))),
                    "ref_rms": float(ref_rms),
                    "relative_rms": float(
                        np.sqrt(np.mean(diff**2)) / ref_rms if ref_rms > 0 else np.inf
                    ),
                    "corr": float(np.corrcoef(ref_sh[finite], comp_sh[finite])[0, 1]),
                }
            )
    results["shear"] = shear_stats

    # Vibration comparison
    n_vib = min(computed.vib.shape[0], reference.vib.shape[0])
    vib_stats = []
    for i in range(n_vib):
        ref_v = reference.vib[i, both_in]
        comp_v = computed.vib[i, both_in]
        finite = np.isfinite(ref_v) & np.isfinite(comp_v)
        if finite.any():
            diff = comp_v[finite] - ref_v[finite]
            ref_rms = np.sqrt(np.mean(ref_v[finite] ** 2))
            vib_stats.append(
                {
                    "sensor": i + 1,
                    "rms_diff": float(np.sqrt(np.mean(diff**2))),
                    "ref_rms": float(ref_rms),
                    "relative_rms": float(
                        np.sqrt(np.mean(diff**2)) / ref_rms if ref_rms > 0 else np.inf
                    ),
                    "corr": float(np.corrcoef(ref_v[finite], comp_v[finite])[0, 1]),
                }
            )
    results["vibration"] = vib_stats

    return results


def format_l2_report(metrics: dict, filename: str = "") -> str:
    """Format L2 comparison metrics as a human-readable report."""
    lines = []
    if filename:
        lines.append(f"L2 comparison: {filename}")
        lines.append("=" * (len(lines[0])))
    else:
        lines.append("L2 comparison")
        lines.append("=============")

    # Section selection
    sec = metrics.get("section", {})
    lines.append("")
    lines.append("Section selection:")
    lines.append(f"  Reference sections:   {sec.get('ref_n_sections', '?')}")
    lines.append(f"  Computed sections:    {sec.get('comp_n_sections', '?')}")
    lines.append(f"  Reference selected:   {sec.get('ref_n_selected', '?')} samples")
    lines.append(f"  Computed selected:    {sec.get('comp_n_selected', '?')} samples")
    lines.append(f"  Overlap fraction:     {sec.get('overlap_frac', 0):.4f}")

    # Speed
    spd = metrics.get("speed")
    if spd:
        lines.append("")
        lines.append("Profiling speed (within overlapping sections):")
        lines.append(f"  Reference mean:       {spd['ref_mean']:.4f} m/s")
        lines.append(f"  Mean difference:      {spd['mean_diff']:.6f} m/s")
        lines.append(f"  RMS difference:       {spd['rms_diff']:.6f} m/s")
        lines.append(f"  Max |difference|:     {spd['max_abs_diff']:.6f} m/s")
        lines.append(f"  Relative RMS:         {spd['relative_rms']:.4f}")

    # Shear
    for sh in metrics.get("shear", []):
        lines.append("")
        lines.append(f"Shear probe {sh['probe']}:")
        lines.append(f"  Reference RMS:        {sh['ref_rms']:.6f} s⁻¹")
        lines.append(f"  RMS difference:       {sh['rms_diff']:.6f} s⁻¹")
        lines.append(f"  Relative RMS diff:    {sh['relative_rms']:.4f}")
        lines.append(f"  Correlation:          {sh['corr']:.6f}")

    # Vibration
    for vb in metrics.get("vibration", []):
        lines.append("")
        lines.append(f"Vibration sensor {vb['sensor']}:")
        lines.append(f"  Reference RMS:        {vb['ref_rms']:.6f}")
        lines.append(f"  RMS difference:       {vb['rms_diff']:.6f}")
        lines.append(f"  Relative RMS diff:    {vb['relative_rms']:.4f}")
        lines.append(f"  Correlation:          {vb['corr']:.6f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# L3 comparison
# ---------------------------------------------------------------------------

def compare_l3(computed: L3Data, reference: L3Data) -> dict:
    """Compare computed L3 spectra against reference L3.

    Compares wavenumber grids, spectral shapes, and per-window metadata.
    """
    results: dict = {}

    results["n_spectra"] = {
        "ref": reference.n_spectra,
        "comp": computed.n_spectra,
    }

    n_common = min(computed.n_spectra, reference.n_spectra)
    if n_common == 0:
        return results

    # Speed comparison
    spd_diff = computed.pspd_rel[:n_common] - reference.pspd_rel[:n_common]
    finite = np.isfinite(spd_diff)
    if finite.any():
        results["speed"] = {
            "rms_diff": float(np.sqrt(np.mean(spd_diff[finite] ** 2))),
            "ref_mean": float(np.nanmean(reference.pspd_rel[:n_common])),
        }

    # Pressure comparison
    pres_diff = computed.pres[:n_common] - reference.pres[:n_common]
    finite = np.isfinite(pres_diff)
    if finite.any():
        results["pressure"] = {
            "rms_diff": float(np.sqrt(np.mean(pres_diff[finite] ** 2))),
            "ref_mean": float(np.nanmean(reference.pres[:n_common])),
        }

    # Spectral comparison — compare log10(spectrum) to handle dynamic range
    n_sh = min(computed.n_shear, reference.n_shear)
    n_wn = min(computed.n_wavenumber, reference.n_wavenumber)

    spec_stats = []
    for label, comp_spec, ref_spec in [
        ("raw", computed.sh_spec, reference.sh_spec),
        ("clean", computed.sh_spec_clean, reference.sh_spec_clean),
    ]:
        for i in range(n_sh):
            # Shape: (n_wavenumber, n_spectra) — compare in log space
            c = comp_spec[i, :n_wn, :n_common]
            r = ref_spec[i, :n_wn, :n_common]

            # Skip DC bin and use only positive values
            valid = (c > 0) & (r > 0)
            if not valid.any():
                continue

            log_c = np.log10(c[valid])
            log_r = np.log10(r[valid])
            log_diff = log_c - log_r
            abs_log_diff = np.abs(log_diff)

            q0 = float(np.min(abs_log_diff))
            q50 = float(np.median(abs_log_diff))
            q100 = float(np.max(abs_log_diff))

            spec_stats.append({
                "type": label,
                "probe": i + 1,
                "mean_log_diff": float(np.mean(log_diff)),
                "rms_log_diff": float(np.sqrt(np.mean(log_diff ** 2))),
                "median_ratio": float(10 ** np.median(log_diff)),
                "q0_abs_log": q0,
                "q50_abs_log": q50,
                "q100_abs_log": q100,
                "n_values": int(valid.sum()),
            })
    results["spectra"] = spec_stats

    return results


def format_l3_report(metrics: dict, filename: str = "") -> str:
    """Format L3 comparison metrics as a human-readable report."""
    lines = []
    header = f"L3 comparison: {filename}" if filename else "L3 comparison"
    lines.append(header)
    lines.append("=" * len(header))

    ns = metrics.get("n_spectra", {})
    lines.append(f"\n  Reference spectra:  {ns.get('ref', '?')}")
    lines.append(f"  Computed spectra:   {ns.get('comp', '?')}")

    spd = metrics.get("speed")
    if spd:
        lines.append(f"\n  Speed RMS diff:     {spd['rms_diff']:.6f} m/s "
                      f"(ref mean {spd['ref_mean']:.4f})")

    prs = metrics.get("pressure")
    if prs:
        lines.append(f"  Pressure RMS diff:  {prs['rms_diff']:.4f} dbar "
                      f"(ref mean {prs['ref_mean']:.1f})")

    for ss in metrics.get("spectra", []):
        lines.append(f"\n  Shear probe {ss['probe']} ({ss['type']}):")
        lines.append(f"    Mean log10 diff:    {ss['mean_log_diff']:+.4f}")
        lines.append(f"    RMS log10 diff:     {ss['rms_log_diff']:.4f}")
        lines.append(f"    Median ratio:       {ss['median_ratio']:.4f}")
        lines.append(
            f"    |log10 diff| Q0/Q50/Q100: "
            f"{ss['q0_abs_log']:.4f} / {ss['q50_abs_log']:.4f} / "
            f"{ss['q100_abs_log']:.4f}"
        )
        lines.append(f"    N values compared:  {ss['n_values']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# L4 comparison
# ---------------------------------------------------------------------------

def compare_l4(computed: L4Data, reference: L4Data) -> dict:
    """Compare computed L4 against reference L4.

    Returns a dict of comparison metrics.
    """
    results: dict = {}

    results["n_spectra"] = {
        "ref": reference.n_spectra,
        "comp": computed.n_spectra,
    }

    n_common = min(computed.n_spectra, reference.n_spectra)
    if n_common == 0:
        return results

    n_sh = min(computed.n_shear, reference.n_shear)

    # Per-probe epsilon comparison (log10 space — epsilon spans decades)
    probe_stats = []
    for i in range(n_sh):
        ref_e = reference.epsi[i, :n_common]
        comp_e = computed.epsi[i, :n_common]
        valid = np.isfinite(ref_e) & np.isfinite(comp_e) & (ref_e > 0) & (comp_e > 0)
        if not valid.any():
            continue

        log_ref = np.log10(ref_e[valid])
        log_comp = np.log10(comp_e[valid])
        log_diff = log_comp - log_ref
        abs_log_diff = np.abs(log_diff)

        probe_stats.append({
            "probe": i + 1,
            "n_valid": int(valid.sum()),
            "mean_log_diff": float(np.mean(log_diff)),
            "rms_log_diff": float(np.sqrt(np.mean(log_diff ** 2))),
            "median_ratio": float(10 ** np.median(log_diff)),
            "q0_abs_log": float(np.min(abs_log_diff)),
            "q25_abs_log": float(np.percentile(abs_log_diff, 25)),
            "q50_abs_log": float(np.median(abs_log_diff)),
            "q75_abs_log": float(np.percentile(abs_log_diff, 75)),
            "q100_abs_log": float(np.max(abs_log_diff)),
            "within_factor2": float(np.mean(abs_log_diff < np.log10(2))),
            "within_factor3": float(np.mean(abs_log_diff < np.log10(3))),
        })
    results["probes"] = probe_stats

    # EPSI_FINAL comparison
    ref_ef = reference.epsi_final[:n_common]
    comp_ef = computed.epsi_final[:n_common]
    valid = np.isfinite(ref_ef) & np.isfinite(comp_ef) & (ref_ef > 0) & (comp_ef > 0)
    if valid.any():
        log_diff = np.log10(comp_ef[valid]) - np.log10(ref_ef[valid])
        abs_log_diff = np.abs(log_diff)
        results["epsi_final"] = {
            "n_valid": int(valid.sum()),
            "mean_log_diff": float(np.mean(log_diff)),
            "rms_log_diff": float(np.sqrt(np.mean(log_diff ** 2))),
            "median_ratio": float(10 ** np.median(log_diff)),
            "q0_abs_log": float(np.min(abs_log_diff)),
            "q50_abs_log": float(np.median(abs_log_diff)),
            "q100_abs_log": float(np.max(abs_log_diff)),
            "within_factor2": float(np.mean(abs_log_diff < np.log10(2))),
        }

    # FOM comparison
    fom_stats = []
    for i in range(n_sh):
        ref_f = reference.fom[i, :n_common]
        comp_f = computed.fom[i, :n_common]
        valid = np.isfinite(ref_f) & np.isfinite(comp_f)
        if valid.any():
            diff = comp_f[valid] - ref_f[valid]
            fom_stats.append({
                "probe": i + 1,
                "ref_mean": float(np.mean(ref_f[valid])),
                "comp_mean": float(np.mean(comp_f[valid])),
                "rms_diff": float(np.sqrt(np.mean(diff ** 2))),
                "corr": float(np.corrcoef(ref_f[valid], comp_f[valid])[0, 1])
                    if len(ref_f[valid]) > 1 else np.nan,
            })
    results["fom"] = fom_stats

    # Method agreement
    method_stats = []
    for i in range(n_sh):
        ref_m = reference.method[i, :n_common]
        comp_m = computed.method[i, :n_common]
        valid = np.isfinite(ref_m) & np.isfinite(comp_m)
        if valid.any():
            agree = (ref_m[valid] == comp_m[valid]).sum()
            method_stats.append({
                "probe": i + 1,
                "agreement": float(agree / valid.sum()),
                "ref_isr_frac": float((ref_m[valid] == 1).mean()),
                "comp_isr_frac": float((comp_m[valid] == 1).mean()),
            })
    results["method"] = method_stats

    return results


def format_l4_report(metrics: dict, filename: str = "") -> str:
    """Format L4 comparison metrics as a human-readable report."""
    lines = []
    header = f"L4 comparison: {filename}" if filename else "L4 comparison"
    lines.append(header)
    lines.append("=" * len(header))

    ns = metrics.get("n_spectra", {})
    lines.append(f"\n  Reference spectra:  {ns.get('ref', '?')}")
    lines.append(f"  Computed spectra:   {ns.get('comp', '?')}")

    for ps in metrics.get("probes", []):
        lines.append(f"\n  Shear probe {ps['probe']}:")
        lines.append(f"    N valid:            {ps['n_valid']}")
        lines.append(f"    Mean log10 diff:    {ps['mean_log_diff']:+.4f}")
        lines.append(f"    RMS log10 diff:     {ps['rms_log_diff']:.4f}")
        lines.append(f"    Median ratio:       {ps['median_ratio']:.4f}")
        lines.append(f"    |log10 diff| Q0/Q25/Q50/Q75/Q100: "
                      f"{ps['q0_abs_log']:.4f} / {ps['q25_abs_log']:.4f} / "
                      f"{ps['q50_abs_log']:.4f} / {ps['q75_abs_log']:.4f} / "
                      f"{ps['q100_abs_log']:.4f}")
        lines.append(f"    Within factor 2:    {ps['within_factor2']:.1%}")
        lines.append(f"    Within factor 3:    {ps['within_factor3']:.1%}")

    ef = metrics.get("epsi_final")
    if ef:
        lines.append("\n  EPSI_FINAL (combined):")
        lines.append(f"    N valid:            {ef['n_valid']}")
        lines.append(f"    Mean log10 diff:    {ef['mean_log_diff']:+.4f}")
        lines.append(f"    RMS log10 diff:     {ef['rms_log_diff']:.4f}")
        lines.append(f"    Median ratio:       {ef['median_ratio']:.4f}")
        lines.append(f"    |log10 diff| Q0/Q50/Q100: "
                      f"{ef['q0_abs_log']:.4f} / {ef['q50_abs_log']:.4f} / "
                      f"{ef['q100_abs_log']:.4f}")
        lines.append(f"    Within factor 2:    {ef['within_factor2']:.1%}")

    for fs in metrics.get("fom", []):
        lines.append(f"\n  FOM probe {fs['probe']}:")
        lines.append(f"    Ref mean: {fs['ref_mean']:.4f}  Comp mean: {fs['comp_mean']:.4f}"
                      f"  RMS diff: {fs['rms_diff']:.4f}  Corr: {fs['corr']:.4f}")

    for ms in metrics.get("method", []):
        lines.append(f"\n  Method probe {ms['probe']}:")
        lines.append(f"    Agreement: {ms['agreement']:.1%}"
                      f"  Ref ISR: {ms['ref_isr_frac']:.1%}"
                      f"  Comp ISR: {ms['comp_isr_frac']:.1%}")

    return "\n".join(lines)
