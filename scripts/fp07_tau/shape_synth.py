"""Does the free-shape fit recover the response exponent, or is it biased toward ~1?"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import medlog
import ratio as R
import ratio_synth as RS
from analyze_cache import summarize

F = RS.F[RS.F <= 100]

def resp(kind, f, p):
    if kind == "sp":    return R.h2(f, p, 1)
    if kind == "dp":    return R.h2(f, p, 2)
    if kind == "a1":    return 1 / (1 + f / p)           # p = fc [Hz], alpha = 1
    raise ValueError(kind)

def run(kind, p1, p2, n=3000, sig_tau=0.0, snr=300, rng=None):
    rng = rng or np.random.default_rng(1)
    S1, S2, N1, N2 = [], [], [], []
    for _ in range(n):
        W = np.exp(rng.normal(0, 0.05)); kB = np.exp(rng.normal(np.log(120), 0.6))
        x = np.maximum(F / W / kB, 1e-6); g = RS.shape(x, rng); lvl = np.exp(rng.normal(0, 1.2))
        nz = np.full(F.size, lvl * np.max(g) / snr) * np.exp(rng.normal(0, 1.0))
        m = np.exp(rng.normal(0, sig_tau))   # window-to-window scatter of the response, common to both beads
        if kind == "a1":
            h1, h2 = resp(kind, F, p1 / m), resp(kind, F, p2 / m)
        else:
            h1, h2 = resp(kind, F, p1 * m), resp(kind, F, p2 * m)
        a = (lvl * g * h1 + nz) * rng.chisquare(RS.DOF, F.size) / RS.DOF
        b = (1.1 * lvl * g * h2 + nz) * rng.chisquare(RS.DOF, F.size) / RS.DOF
        S1.append(a); S2.append(b); N1.append(nz); N2.append(nz)
    lr, _ = medlog.ratio_curve(*map(np.array, (S1, S2, N1, N2)))
    return summarize(F, lr)

cases = [("sp", 0.015, 0.010, 0.0), ("sp", 0.020, 0.008, 0.0), ("dp", 0.008, 0.005, 0.0),
         ("a1", 21.0, 5.1, 0.0), ("a1", 32.0, 107.0, 0.0),
         ("sp", 0.015, 0.010, 0.3), ("sp", 0.020, 0.008, 0.5)]
if "mixed" not in sys.argv:
  for kind, p1, p2, st in cases:
      s = run(kind, p1, p2, sig_tau=st)
      unit = "ms" if kind != "a1" else "Hz"
      tv = (p1 * 1e3, p2 * 1e3) if kind != "a1" else (p1, p2)
      print(f"truth {kind} {tv[0]:.0f}/{tv[1]:.0f} {unit} scatter {st}: free alpha {s['fs_alpha']:.2f} fc {s['fs_fc1']:.1f}/{s['fs_fc2']:.1f} Hz "
            f"(rms free {s['fs_rms']:.4f} SP {s['full_p1_rms']:.4f} DP {s['full_p2_rms']:.4f})  SP25 {s['b25_p1_tau1']:.1f}/{s['b25_p1_tau2']:.1f}  SP70 {s['full_p1_tau1']:.1f}/{s['full_p1_tau2']:.1f} ms")


def h2_half(f, tau):
    """|H|^2 for H = 1/(1 + sqrt(i w tau)): diffusion through a layer into a core."""
    s = np.sqrt(2 * np.pi * f * tau / 2)
    return 1.0 / ((1 + s) ** 2 + s ** 2)


def run_mixed(h1f, h2f, n=3000, snr=300, rng=None):
    rng = rng or np.random.default_rng(2)
    S1, S2, N1, N2 = [], [], [], []
    for _ in range(n):
        W = np.exp(rng.normal(0, 0.05)); kB = np.exp(rng.normal(np.log(120), 0.6))
        x = np.maximum(F / W / kB, 1e-6); g = RS.shape(x, rng); lvl = np.exp(rng.normal(0, 1.2))
        nz = np.full(F.size, lvl * np.max(g) / snr) * np.exp(rng.normal(0, 1.0))
        a = (lvl * g * h1f(F) + nz) * rng.chisquare(RS.DOF, F.size) / RS.DOF
        b = (1.1 * lvl * g * h2f(F) + nz) * rng.chisquare(RS.DOF, F.size) / RS.DOF
        S1.append(a); S2.append(b); N1.append(nz); N2.append(nz)
    lr, _ = medlog.ratio_curve(*map(np.array, (S1, S2, N1, N2)))
    return summarize(F, lr)


if __name__ == "__main__" and "mixed" in sys.argv:
    for label, h1f, h2f in [
        ("SP 15 ms / DP 5 ms", lambda f: R.h2(f, 0.015, 1), lambda f: R.h2(f, 0.005, 2)),
        ("DP 8 ms / SP 8 ms", lambda f: R.h2(f, 0.008, 2), lambda f: R.h2(f, 0.008, 1)),
        ("half-order 20 ms / 120 ms", lambda f: h2_half(f, 0.020), lambda f: h2_half(f, 0.120)),
        ("half-order 5 ms / 40 ms", lambda f: h2_half(f, 0.005), lambda f: h2_half(f, 0.040)),
    ]:
        s = run_mixed(h1f, h2f)
        print(f"{label:26s}: free alpha {s['fs_alpha']:.2f} fc {s['fs_fc1']:.1f}/{s['fs_fc2']:.1f}  rms free {s['fs_rms']:.4f} SP {s['full_p1_rms']:.4f}  SP25 {s['b25_p1_tau1']:.1f}/{s['b25_p1_tau2']:.1f} SP70 {s['full_p1_tau1']:.1f}/{s['full_p1_tau2']:.1f} ms")
