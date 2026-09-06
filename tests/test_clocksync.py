# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for clock synchronization.

The load-bearing test is :func:`test_sign_and_accuracy`: it injects a known
shift and asserts the recovered value, sign included.  A sign error here is
invisible in every summary statistic and would corrupt every downstream
result, so it is pinned rather than reasoned about.
"""

from __future__ import annotations

import numpy as np
import pytest

from odas_tpw.clocksync.fit import fit_file
from odas_tpw.clocksync.lag import bandpass, estimate_lag, shift
from odas_tpw.clocksync.qc import check_continuity, summarize
from odas_tpw.clocksync.reference import to_epoch
from odas_tpw.clocksync.series import PressureSeries, epoch_from_datenum

FS = 16.0


def synthetic_sea(n: int, seed: int = 0, f0: float = 0.12, width: float = 0.035,
                  amp: float = 0.16) -> np.ndarray:
    """A random sea with a realistic peak: amplitude in dbar, like the bank."""
    f = np.fft.rfftfreq(n, 1.0 / FS)
    S = np.exp(-(((f - f0) / width) ** 2))
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, f.size)
    x = np.fft.irfft(np.sqrt(S) * np.exp(1j * ph), n)
    return amp * x / x.std()


def tide(n: int, amp: float = 0.9) -> np.ndarray:
    t = np.arange(n) / FS
    return amp * np.sin(2 * np.pi * t / 44700.0)


@pytest.mark.parametrize("true_lag", [0.0, 0.0625, 0.31, -1.7, 5.0, -12.3])
def test_sign_and_accuracy(true_lag):
    """Recovered lag equals the injected shift, sign included."""
    n = int(FS * 900)
    rng = np.random.default_rng(4)
    wave = synthetic_sea(n, seed=1)
    ref = wave + tide(n) + 0.002 * rng.standard_normal(n)
    # The target's clock is SLOW by true_lag, so its record appears shifted
    # earlier; adding true_lag to its timestamps is the correction.
    tgt = shift(wave, FS, -true_lag) + tide(n) + 0.002 * rng.standard_normal(n)
    r = estimate_lag(ref, tgt, FS, band=(0.05, 0.35), max_lag=60.0)
    assert r.ok, r.why
    assert r.lag == pytest.approx(true_lag, abs=0.01)
    assert r.sigma < 0.01


def test_sigma_is_honest():
    """The quoted sigma brackets the real error as noise is added."""
    n = int(FS * 900)
    wave = synthetic_sea(n, seed=2)
    errs, sigs = [], []
    for i, noise in enumerate([0.005, 0.02, 0.05]):
        rng = np.random.default_rng(100 + i)
        ref = wave + noise * rng.standard_normal(n)
        tgt = shift(wave, FS, -0.75) + noise * rng.standard_normal(n)
        r = estimate_lag(ref, tgt, FS, max_lag=60.0)
        assert r.ok, r.why
        errs.append(abs(r.lag - 0.75))
        sigs.append(r.sigma)
    assert sigs[0] < sigs[-1], "sigma must grow with noise"
    assert all(e < 6 * s for e, s in zip(errs, sigs, strict=True))


def test_tide_alone_is_rejected():
    """A window holding only tide has no wave, and must not yield a lag.

    Without the amplitude gate the band-pass leaves numerical ringing that is
    identical in both records: coherence 1.00, a sharp peak, and a confident
    lag derived from filter transients.
    """
    n = int(FS * 900)
    t = np.arange(n) / FS
    a = np.sin(2 * np.pi * t / 44700.0)
    b = np.sin(2 * np.pi * (t - 3.0) / 44700.0)
    r = estimate_lag(a, b, FS, max_lag=60.0)
    assert not r.ok
    assert "wave energy" in r.why


def test_uncorrelated_is_rejected():
    rng = np.random.default_rng(5)
    n = int(FS * 900)
    r = estimate_lag(rng.standard_normal(n), rng.standard_normal(n), FS, max_lag=60.0)
    assert not r.ok


def test_bandpass_is_zero_phase():
    """A causal filter would inject its own group delay into the answer."""
    n = int(FS * 600)
    x = synthetic_sea(n, seed=3)
    y = bandpass(x, FS, 0.05, 0.35)
    r = estimate_lag(x, y, FS, max_lag=30.0)
    assert r.ok, r.why
    assert abs(r.lag) < 0.01


def test_shift_roundtrip():
    n = 4096
    x = synthetic_sea(n, seed=6)
    back = shift(shift(x, FS, 0.37), FS, -0.37)
    assert np.allclose(x[100:-100], back[100:-100], atol=1e-6)


def test_fit_file_recovers_offset_and_rate():
    """A file whose clock is offset AND drifting is recovered in both terms."""
    n = int(FS * 3600 * 3)
    t0 = 1.7e9
    wave = synthetic_sea(n, seed=7) + tide(n)
    ref = PressureSeries(t0 + np.arange(n) / FS, wave, "ref", fs=FS)
    offset, rate = -4.25, 3.0e-5          # 2.6 s/day
    # Build the target by sampling the same field on a clock that is wrong.
    tt = t0 + np.arange(n) / FS
    true_t = tt + offset + rate * (tt - t0)
    tgt = PressureSeries(tt, np.interp(true_t, ref.t, ref.p), "tgt", fs=FS)
    f = fit_file(ref, tgt, FS, window=900.0, band=(0.05, 0.35), max_lag=60.0)
    assert f.ok, f.why
    assert f.n_used >= 6
    assert f.offset == pytest.approx(offset, abs=0.05)
    assert f.rate == pytest.approx(rate, rel=0.3)
    assert f.correction(np.array([t0]))[0] == pytest.approx(offset, abs=0.05)


def test_continuity_names_a_cycle_slip():
    from odas_tpw.clocksync.fit import FileFit

    fits = []
    for i in range(21):
        fits.append(FileFit(name=f"f{i:02d}", t0=1.7e9 + i * 10840.0,
                            offset=1.0 + 0.001 * i, offset_sigma=0.01,
                            rate=0.0, ok=True, n_used=8, coherence=0.9))
    fits[10].offset += 8.3                                   # exactly one period
    check_continuity(fits, n_sigma=5.0, window=9, wave_period=8.3)
    assert any("CYCLE SLIP" in m for m in fits[10].flags)
    assert not fits[9].flags and not fits[11].flags
    s = summarize(fits)
    assert s["n_solved"] == 21 and s["n_flagged"] == 1


def test_datenum_and_units():
    dn = 739013.1666666782                     # 2023-05-07 04:00 UTC
    assert epoch_from_datenum(np.array([dn]))[0] == pytest.approx(1683432000.0, abs=1.0)
    _, kind = to_epoch(np.array([dn]))
    assert kind == "datenum"
    _, kind = to_epoch(np.array([1.7e9]))
    assert kind == "epoch"
    with pytest.raises(ValueError):
        to_epoch(np.array([12.0]))


def test_uniform_blanks_gaps_not_bridges_them():
    t = np.concatenate([np.arange(0, 100, 0.5), np.arange(400, 500, 0.5)]) + 1.7e9
    s = PressureSeries(t, np.sin(t), "s", fs=2.0)
    u = s.uniform(2.0, max_gap=1.0)
    mid = (u.t > 1.7e9 + 150) & (u.t < 1.7e9 + 350)
    assert np.isnan(u.p[mid]).all(), "the hole must stay a hole"
    assert np.isfinite(u.p[u.t < 1.7e9 + 99]).all()


def test_time_stays_float64():
    """float32 epoch seconds resolve to ~128 s and destroy every dt."""
    t = 1.7e9 + np.arange(1000) / 16.0
    s = PressureSeries(t.astype(np.float32), np.zeros(1000), "s")
    assert s.t.dtype == np.float64
    # The guard is the dtype; a float32 round-trip loses the sample spacing.
    assert np.median(np.diff(t.astype(np.float32).astype(np.float64))) != pytest.approx(
        1 / 16.0, rel=0.1
    )


def test_cache_roundtrip_and_reuse(tmp_path):
    """The cache entry lands at the path we asked for, and is reused.

    np.savez_compressed appends ".npz" to any path not already ending in it,
    which once made the atomic rename fail on a file that had been written
    correctly.  Both halves are pinned here.
    """
    import numpy as np

    from odas_tpw.clocksync.extract import CACHE_VERSION, cache_path, load_cached

    src = tmp_path / "fake_0001.p"
    src.write_bytes(b"not a real p file")
    out = cache_path(tmp_path / "cache", src)
    (tmp_path / "cache").mkdir()
    n, fs, t0 = 512, 16.0, 1.7e9
    p = synthetic_sea(n, seed=11)
    with open(out.with_suffix(".tmp.npz"), "wb") as fh:
        np.savez_compressed(fh, version=CACHE_VERSION, t0=t0, fs=fs,
                            p=p.astype(np.float32), mtime=0.0, source=str(src))
    out.with_suffix(".tmp.npz").replace(out)
    assert out.exists() and not out.with_suffix(".tmp.npz.npz").exists()
    s = load_cached(out)
    assert s.p.size == n
    assert s.t.dtype == np.float64
    assert s.t[0] == pytest.approx(t0)
    assert np.median(np.diff(s.t)) == pytest.approx(1 / fs)
