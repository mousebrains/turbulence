# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Branch-coverage tests for perturb.ctd — narrow time spans, fast-only, missing seawater."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from odas_tpw.perturb.ctd import ctd_bin_file
from odas_tpw.perturb.gps import GPSFixed


class _PFileStub:
    def __init__(
        self,
        channels,
        t_slow,
        t_fast=None,
        fs_slow=64.0,
        fs_fast=512.0,
        filepath=None,
        fast_channels=None,
    ):
        self.channels = channels
        self.t_slow = t_slow
        self.t_fast = t_fast if t_fast is not None else t_slow
        self.fs_slow = fs_slow
        self.fs_fast = fs_fast
        self.filepath = filepath or Path("test_001.p")
        self._fast = fast_channels or set()

    def is_fast(self, ch):
        return ch in self._fast


# ---------------------------------------------------------------------------
# Tests for narrow time spans / fast-only / missing seawater inputs
# ---------------------------------------------------------------------------


class TestCtdBinFileBranches:
    def test_narrow_span_forces_minimum_two_edges(self, tmp_path):
        """When t_hi - t_lo < bin_width → bin_edges falls back to [t_lo, t_lo+bw]."""
        # Two slow samples spaced by 0.001s, bin_width=10s → only 1 edge from arange
        t_slow = np.array([0.0, 0.001])
        channels = {
            "JAC_T": np.array([10.0, 10.1]),
            "JAC_C": np.array([33.0, 33.1]),
            "P": np.array([0.0, 0.001]),
        }
        pf = _PFileStub(channels=channels, t_slow=t_slow)
        gps = GPSFixed(15.0, 145.0)
        # bin_width=10 forces the (t_hi - t_lo + bw) range to span ~10s,
        # producing arange of just 1 element → fallback path
        result = ctd_bin_file(pf, gps, tmp_path, bin_width=10.0)
        assert result is not None  # didn't crash; produced output

    def test_only_fast_channels_uses_fast_bin_centers(self, tmp_path):
        """No slow channels at all → fast branch sets bin_centers (line 191 area)."""
        n_fast = 200
        t_fast = np.linspace(0, 4.0, n_fast)
        # No JAC slow channels — only fast channels
        # Note: .stuff like P and JAC_T are typically slow; here we mark them fast
        channels = {
            "JAC_T": np.full(n_fast, 10.0),
            "JAC_C": np.full(n_fast, 33.0),
            "P": np.full(n_fast, 50.0),
        }
        pf = _PFileStub(
            channels=channels,
            t_slow=t_fast,  # placeholder
            t_fast=t_fast,
            fast_channels={"JAC_T", "JAC_C", "P"},  # mark all as fast
        )
        gps = GPSFixed(15.0, 145.0)
        result = ctd_bin_file(pf, gps, tmp_path, bin_width=0.5)
        # Fast-only path should still produce output
        assert result is not None

    def test_no_temperature_no_seawater_props(self, tmp_path):
        """When T_name not in binned → skip seawater-property block (line 209 false branch)."""
        # Provide only P and a custom channel — no JAC_T/JAC_C
        n = 200
        t_slow = np.linspace(0, 4.0, n)
        channels = {
            "P": np.linspace(0, 50, n),
            # No JAC_T — so seawater computation is skipped
        }
        pf = _PFileStub(channels=channels, t_slow=t_slow)
        gps = GPSFixed(15.0, 145.0)
        # Pass channels=["P"] explicitly via the variables dict
        result = ctd_bin_file(
            pf, gps, tmp_path, bin_width=0.5, variables={"channels": ["P"]}
        )
        assert result is not None
        # Should NOT contain SP/SA/CT (seawater props) since T/C absent
        import xarray as xr

        ds = xr.open_dataset(result)
        assert "SP" not in ds
        assert "SA" not in ds
        ds.close()

    def test_no_channels_at_all_returns_none(self, tmp_path):
        """No matching channels → returns None (line 197 area)."""
        n = 64
        t_slow = np.linspace(0, 1, n)
        # Channels not in the default JAC list and no P
        channels = {"Foo": np.zeros(n), "Bar": np.zeros(n)}
        pf = _PFileStub(channels=channels, t_slow=t_slow)
        gps = GPSFixed(0.0, 0.0)
        result = ctd_bin_file(pf, gps, tmp_path)
        assert result is None
