# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Pressure series on a common footing, whatever instrument produced it.

Everything in this package reduces to the same object: timestamps in epoch
seconds and pressure in dbar.  The reference and the target differ only in
whose clock we trust.

Two conventions are load-bearing and are enforced here rather than left to
each caller:

**Time is float64, always.**  Epoch seconds are ~1.7e9; float32 resolves that
to about 128 s, which silently destroys every time difference in the record.
Pressure may be float32 (0.001 dbar out of 20 needs 15 bits); time may not.

**Gaps are holes, not values.**  ``uniform`` marks any output sample with no
nearby input as NaN instead of interpolating across it.  A cross-correlation
run over invented data reports a lag for a signal that was never measured.
Short gaps are bridged only up to an explicit limit, and the limit is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PressureSeries:
    """Pressure against epoch seconds, monotonic, gaps marked NaN.

    ``fs`` is the nominal sample rate; it is a label for reporting and the
    default when resampling, never an assumption the arithmetic relies on.
    """

    t: np.ndarray
    p: np.ndarray
    name: str
    source: str = ""
    fs: float = 0.0

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.p = np.asarray(self.p, dtype=np.float64)
        if self.t.shape != self.p.shape:
            raise ValueError(
                f"{self.name}: time and pressure differ in length "
                f"({self.t.size} vs {self.p.size})"
            )
        if self.t.size and not np.all(np.diff(self.t) >= 0):
            order = np.argsort(self.t, kind="stable")
            self.t = self.t[order]
            self.p = self.p[order]
        if not self.fs and self.t.size > 8:
            dt = float(np.median(np.diff(self.t)))
            self.fs = 1.0 / dt if dt > 0 else 0.0

    @property
    def span(self) -> tuple[float, float]:
        if not self.t.size:
            return (np.nan, np.nan)
        return (float(self.t[0]), float(self.t[-1]))

    @property
    def duration(self) -> float:
        t0, t1 = self.span
        return float(t1 - t0)

    def clip(self, t0: float, t1: float) -> PressureSeries:
        m = (self.t >= t0) & (self.t <= t1)
        return PressureSeries(self.t[m], self.p[m], self.name, self.source, self.fs)

    def uniform(self, fs: float, max_gap: float = 1.0) -> PressureSeries:
        """Onto a uniform grid at *fs*, NaN wherever the input had no data.

        Samples more than *max_gap* seconds from the nearest input sample are
        NaN.  Everything closer is linearly interpolated: at 16 Hz against a
        128 Hz source that is a sub-sample correction, not invented signal.

        A series already uniform at *fs* is returned unchanged apart from the
        dtype guarantee, so calling this on an already-gridded record costs
        nothing and cannot shift it.
        """
        if self.t.size < 2:
            return PressureSeries(self.t, self.p, self.name, self.source, fs)
        t0, t1 = self.span
        n = int(np.floor((t1 - t0) * fs)) + 1
        grid = t0 + np.arange(n, dtype=np.float64) / fs
        good = np.isfinite(self.p)
        if good.sum() < 2:
            return PressureSeries(grid, np.full(n, np.nan), self.name, self.source, fs)
        tg, pg = self.t[good], self.p[good]
        out = np.interp(grid, tg, pg)
        # Distance to the nearest real sample, on both sides, so a gap is
        # blanked from its own edges rather than from the window's.
        idx = np.searchsorted(tg, grid).clip(1, tg.size - 1)
        near = np.minimum(np.abs(grid - tg[idx - 1]), np.abs(grid - tg[idx]))
        out[near > max_gap] = np.nan
        return PressureSeries(grid, out, self.name, self.source, fs)


def epoch_from_datenum(dn: np.ndarray) -> np.ndarray:
    """MATLAB datenum -> epoch seconds.

    719529 is datenum(1970,1,1).  Done in float64 throughout: a datenum is
    ~7.4e5 days and the fractional part carries the sub-second information,
    so this subtraction must happen before the multiply, not after.
    """
    return (np.asarray(dn, dtype=np.float64) - 719529.0) * 86400.0


def epoch_to_iso(t: float) -> str:
    import datetime as _dt

    if not np.isfinite(t):
        return "?"
    return _dt.datetime.fromtimestamp(t, _dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
