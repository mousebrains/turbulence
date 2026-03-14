# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.seawater — seawater properties."""

import numpy as np

from microstructure_tpw.perturb.seawater import add_seawater_properties


class TestAddSeawaterProperties:
    def test_standard_ocean(self):
        """Standard ocean conditions: T=10, C~37 mS/cm, P=100 dbar."""
        T = np.array([10.0])
        C = np.array([37.0])  # approximate for S~35
        P = np.array([100.0])
        lat = np.array([15.0])
        lon = np.array([145.0])

        props = add_seawater_properties(T, C, P, lat, lon)

        assert "SP" in props
        assert "SA" in props
        assert "CT" in props
        assert "sigma0" in props
        assert "rho" in props
        assert "depth" in props

        # Practical salinity should be reasonable (20-40 PSU)
        assert 20.0 < props["SP"][0] < 40.0
        # Depth should be positive and reasonable
        assert props["depth"][0] > 0
        assert props["depth"][0] < 200.0

    def test_nan_lat_lon(self):
        """NaN lat/lon should not cause errors (replaced with 0)."""
        T = np.array([15.0])
        C = np.array([37.0])
        P = np.array([50.0])
        lat = np.array([np.nan])
        lon = np.array([np.nan])

        props = add_seawater_properties(T, C, P, lat, lon)
        assert np.isfinite(props["depth"][0])

    def test_output_shapes_match(self):
        n = 10
        T = np.linspace(5, 25, n)
        C = np.full(n, 37.0)
        P = np.linspace(0, 200, n)
        lat = np.full(n, 15.0)
        lon = np.full(n, 145.0)

        props = add_seawater_properties(T, C, P, lat, lon)
        for key, val in props.items():
            assert val.shape == (n,), f"{key} shape mismatch"

    def test_depth_increases_with_pressure(self):
        T = np.full(5, 10.0)
        C = np.full(5, 37.0)
        P = np.array([10, 50, 100, 500, 1000], dtype=float)
        lat = np.full(5, 0.0)
        lon = np.full(5, 0.0)

        props = add_seawater_properties(T, C, P, lat, lon)
        assert np.all(np.diff(props["depth"]) > 0)
