# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.hotel — hotel file loading and interpolation."""

from datetime import datetime, timezone

import numpy as np
import pytest

from perturb.hotel import HotelData, interpolate_hotel, load_hotel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(path, time_col="time", extra_cols=None):
    """Write a simple CSV hotel file."""
    import pandas as pd

    data = {time_col: [0, 1, 2, 3, 4]}
    defaults = {"speed": [0.5, 0.6, 0.7, 0.8, 0.9], "pitch": [1, 2, 3, 4, 5]}
    if extra_cols:
        defaults.update(extra_cols)
    data.update(defaults)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _make_nc(path, time_var="time"):
    """Write a simple NetCDF hotel file."""
    import netCDF4 as nc

    ds = nc.Dataset(str(path), "w")
    ds.createDimension("obs", 5)
    t = ds.createVariable(time_var, "f8", ("obs",))
    s = ds.createVariable("speed", "f8", ("obs",))
    p = ds.createVariable("pitch", "f8", ("obs",))
    t[:] = [0, 1, 2, 3, 4]
    s[:] = [0.5, 0.6, 0.7, 0.8, 0.9]
    p[:] = [1, 2, 3, 4, 5]
    ds.close()
    return path


def _make_mat(path):
    """Write a simple .mat hotel file with flat arrays."""
    from scipy.io import savemat

    savemat(str(path), {
        "time": np.array([0, 1, 2, 3, 4], dtype=np.float64),
        "speed": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        "pitch": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    })
    return path


class _MockPFile:
    """Minimal PFile-like object for interpolation tests."""

    def __init__(self, n_fast=50, n_slow=5, start_epoch=1000.0):
        self.fs_fast = 512
        self.fs_slow = 64
        self.t_fast = np.linspace(0, 4, n_fast)
        self.t_slow = np.linspace(0, 4, n_slow)
        self.start_time = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
        self.channels = {}


# ---------------------------------------------------------------------------
# TestLoadHotel
# ---------------------------------------------------------------------------


class TestLoadHotel:
    def test_csv(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file)
        assert "speed" in hd.channels
        assert "pitch" in hd.channels
        assert len(hd.time) == 5
        np.testing.assert_allclose(hd.channels["speed"], [0.5, 0.6, 0.7, 0.8, 0.9])

    def test_netcdf(self, tmp_path):
        nc_file = _make_nc(tmp_path / "hotel.nc")
        hd = load_hotel(nc_file)
        assert "speed" in hd.channels
        assert "pitch" in hd.channels
        assert len(hd.time) == 5

    def test_mat(self, tmp_path):
        mat_file = _make_mat(tmp_path / "hotel.mat")
        hd = load_hotel(mat_file)
        assert "speed" in hd.channels
        assert "pitch" in hd.channels
        assert len(hd.time) == 5

    def test_channel_mapping(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file, channels={"speed": "W"})
        assert "W" in hd.channels
        assert "speed" not in hd.channels
        assert "pitch" not in hd.channels  # filtered out

    def test_unsupported_format_raises(self, tmp_path):
        bad_file = tmp_path / "hotel.xyz"
        bad_file.write_text("data")
        with pytest.raises(ValueError, match="Unsupported hotel file format"):
            load_hotel(bad_file)

    def test_custom_time_column(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv", time_col="epoch")
        hd = load_hotel(csv_file, time_column="epoch")
        assert len(hd.time) == 5


# ---------------------------------------------------------------------------
# TestTimeConversion
# ---------------------------------------------------------------------------


class TestTimeConversion:
    def test_auto_seconds(self, tmp_path):
        """Small values → detected as relative seconds."""
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file, time_format="auto")
        assert hd.time_is_relative is True

    def test_auto_epoch(self, tmp_path):
        import pandas as pd

        csv_file = tmp_path / "hotel.csv"
        epoch = 1700000000.0
        pd.DataFrame({
            "time": [epoch, epoch + 1, epoch + 2],
            "speed": [0.5, 0.6, 0.7],
        }).to_csv(csv_file, index=False)
        hd = load_hotel(csv_file, time_format="auto")
        assert hd.time_is_relative is False
        np.testing.assert_allclose(hd.time[0], epoch)

    def test_explicit_seconds(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file, time_format="seconds")
        assert hd.time_is_relative is True

    def test_explicit_epoch(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file, time_format="epoch")
        assert hd.time_is_relative is False

    def test_iso_format(self, tmp_path):
        import pandas as pd

        csv_file = tmp_path / "hotel.csv"
        pd.DataFrame({
            "time": ["2025-01-01T00:00:00", "2025-01-01T00:00:01", "2025-01-01T00:00:02"],
            "speed": [0.5, 0.6, 0.7],
        }).to_csv(csv_file, index=False)
        hd = load_hotel(csv_file, time_format="iso")
        assert hd.time_is_relative is False
        assert hd.time[1] > hd.time[0]

    def test_relative_time_offset(self, tmp_path):
        """Relative seconds should be used as-is during interpolation."""
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file, time_format="seconds")
        pf = _MockPFile()
        result = interpolate_hotel(hd, pf, {"fast_channels": ["speed"]})
        assert "speed" in result
        assert len(result["speed"]) == len(pf.t_fast)

    def test_epoch_time_offset(self, tmp_path):
        """Epoch seconds should be offset by pf.start_time.timestamp()."""
        import pandas as pd

        start_epoch = 1000.0
        csv_file = tmp_path / "hotel.csv"
        pd.DataFrame({
            "time": [start_epoch, start_epoch + 1, start_epoch + 2,
                     start_epoch + 3, start_epoch + 4],
            "speed": [0.5, 0.6, 0.7, 0.8, 0.9],
        }).to_csv(csv_file, index=False)

        hd = load_hotel(csv_file, time_format="epoch")
        pf = _MockPFile(start_epoch=start_epoch)
        result = interpolate_hotel(hd, pf, {"fast_channels": ["speed"]})
        # At t_fast edges (0 and 4), should match boundary values
        np.testing.assert_allclose(result["speed"][0], 0.5, atol=0.01)
        np.testing.assert_allclose(result["speed"][-1], 0.9, atol=0.01)


# ---------------------------------------------------------------------------
# TestInterpolateHotel
# ---------------------------------------------------------------------------


class TestInterpolateHotel:
    def test_fast_slow_routing(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file, time_format="seconds")
        pf = _MockPFile(n_fast=50, n_slow=5)

        result = interpolate_hotel(hd, pf, {"fast_channels": ["speed"]})
        assert result["speed"].shape == (50,)  # fast axis
        assert result["pitch"].shape == (5,)   # slow axis

    def test_pchip_interpolation(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file, time_format="seconds")
        pf = _MockPFile(n_fast=50, n_slow=5)

        result = interpolate_hotel(hd, pf, {
            "fast_channels": ["speed"],
            "interpolation": "pchip",
        })
        # Pchip should produce smooth values within the data range
        assert np.all(np.isfinite(result["speed"]))
        assert result["speed"].min() >= 0.49  # near boundary

    def test_linear_interpolation(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file, time_format="seconds")
        pf = _MockPFile(n_fast=50, n_slow=5)

        result = interpolate_hotel(hd, pf, {
            "fast_channels": ["speed"],
            "interpolation": "linear",
        })
        assert np.all(np.isfinite(result["speed"]))
        # Linear at t=0.5 should be ~0.55
        idx = np.argmin(np.abs(pf.t_fast - 0.5))
        np.testing.assert_allclose(result["speed"][idx], 0.55, atol=0.02)

    def test_edge_extrapolation(self, tmp_path):
        """When hotel range is shorter, boundary values should be used."""
        import pandas as pd

        csv_file = tmp_path / "hotel.csv"
        pd.DataFrame({
            "time": [1, 2, 3],
            "speed": [0.6, 0.7, 0.8],
        }).to_csv(csv_file, index=False)

        hd = load_hotel(csv_file, time_format="seconds")
        pf = _MockPFile(n_fast=50)  # t_fast spans 0..4

        result = interpolate_hotel(hd, pf, {
            "fast_channels": ["speed"],
            "interpolation": "pchip",
        })
        # Values outside [1, 3] should be filled with boundary values
        before = pf.t_fast < 1.0
        after = pf.t_fast > 3.0
        np.testing.assert_allclose(result["speed"][before], 0.6)
        np.testing.assert_allclose(result["speed"][after], 0.8)

    def test_edge_extrapolation_linear(self, tmp_path):
        """Linear interpolation also fills edges."""
        import pandas as pd

        csv_file = tmp_path / "hotel.csv"
        pd.DataFrame({
            "time": [1, 2, 3],
            "speed": [0.6, 0.7, 0.8],
        }).to_csv(csv_file, index=False)

        hd = load_hotel(csv_file, time_format="seconds")
        pf = _MockPFile(n_fast=50)

        result = interpolate_hotel(hd, pf, {
            "fast_channels": ["speed"],
            "interpolation": "linear",
        })
        before = pf.t_fast < 1.0
        after = pf.t_fast > 3.0
        np.testing.assert_allclose(result["speed"][before], 0.6)
        np.testing.assert_allclose(result["speed"][after], 0.8)

    def test_all_channels_injected(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file, time_format="seconds")
        pf = _MockPFile()

        result = interpolate_hotel(hd, pf, {"fast_channels": []})
        # All channels should go to slow axis when none are in fast_channels
        for name, arr in result.items():
            assert arr.shape == pf.t_slow.shape


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_channels_loads_all(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv")
        hd = load_hotel(csv_file, channels={})
        assert "speed" in hd.channels
        assert "pitch" in hd.channels

    def test_unknown_format_raises(self, tmp_path):
        bad_file = tmp_path / "hotel.txt"
        bad_file.write_text("data")
        with pytest.raises(ValueError, match="Unsupported hotel file format"):
            load_hotel(bad_file)

    def test_unknown_time_format_raises(self, tmp_path):
        csv_file = _make_csv(tmp_path / "hotel.csv")
        with pytest.raises(ValueError, match="Unknown time_format"):
            load_hotel(csv_file, time_format="invalid")

    def test_hotel_data_dataclass(self):
        hd = HotelData(time=np.array([0, 1, 2.0]))
        assert len(hd.time) == 3
        assert hd.channels == {}
        assert hd.time_is_relative is False
