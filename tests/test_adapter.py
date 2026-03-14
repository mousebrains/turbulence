# Tests for odas_tpw.rsi.adapter
"""Unit tests for pfile_to_l1data adapter (PFile -> scor160 L1Data)."""

from datetime import datetime

import numpy as np
import pytest

from odas_tpw.rsi.adapter import pfile_to_l1data

# ---------------------------------------------------------------------------
# Mock PFile
# ---------------------------------------------------------------------------


class MockPFile:
    """Minimal PFile stub providing only the attributes accessed by the adapter."""

    def __init__(
        self,
        *,
        n_slow=500,
        fs_fast=512.0,
        fs_slow=64.0,
        shear_names=("sh1", "sh2"),
        acc_names=("Ax", "Az"),
        piezo_names=(),
        temp_fast_names=(),
        has_T1=True,
        pressure_ramp=(5.0, 50.0),
        start_year=2025,
    ):
        ratio = round(fs_fast / fs_slow)
        n_fast = n_slow * ratio

        self.fs_fast = fs_fast
        self.fs_slow = fs_slow
        self.t_slow = np.arange(n_slow) / fs_slow
        self.t_fast = np.arange(n_fast) / fs_fast
        self.start_time = datetime(start_year, 1, 15)

        # Build channel data and metadata
        rng = np.random.default_rng(99)
        self.channels = {}
        self.channel_info = {}
        self._fast_channels = set()

        # Pressure (slow)
        self.channels["P"] = np.linspace(pressure_ramp[0], pressure_ramp[1], n_slow)
        self.channel_info["P"] = {"type": "pres", "units": "dbar"}

        # Temperature (slow)
        if has_T1:
            self.channels["T1"] = np.linspace(20.0, 5.0, n_slow)
            self.channel_info["T1"] = {"type": "therm", "units": "degC"}

        # Shear probes (fast)
        for name in shear_names:
            self.channels[name] = rng.standard_normal(n_fast) * 0.1
            self.channel_info[name] = {"type": "shear", "units": "s-1"}
            self._fast_channels.add(name)

        # Accelerometers (fast)
        for name in acc_names:
            self.channels[name] = rng.standard_normal(n_fast) * 0.01
            self.channel_info[name] = {"type": "accel", "units": "m/s2"}
            self._fast_channels.add(name)

        # Piezo vibration sensors (fast)
        for name in piezo_names:
            self.channels[name] = rng.standard_normal(n_fast) * 0.005
            self.channel_info[name] = {"type": "piezo", "units": "V"}
            self._fast_channels.add(name)

        # Fast thermistors (T1_dT1, etc.)
        for name in temp_fast_names:
            self.channels[name] = np.linspace(20.0, 5.0, n_fast)
            self.channel_info[name] = {"type": "therm", "units": "degC"}
            self._fast_channels.add(name)

        # Config dict (instrument_info, channels list)
        self.config = {
            "instrument_info": {"vehicle": "VMP"},
            "channels": [{"name": n, "id": i} for i, n in enumerate(self.channels)],
        }
        # Add diff_gain entries for fast thermistor channels
        for ch in self.config["channels"]:
            if ch["name"] in temp_fast_names:
                ch["diff_gain"] = "0.94"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicConversion:
    """Mock PFile with shear, pressure, accelerometers -> L1Data."""

    def test_fields_populated(self):
        pf = MockPFile()
        l1 = pfile_to_l1data(pf, speed=0.6)

        assert l1.n_shear == 2
        assert l1.n_vib == 2
        assert l1.vib_type == "ACC"
        assert l1.fs_fast == 512.0
        assert l1.fs_slow == 64.0
        assert l1.f_AA == 98.0
        assert l1.vehicle == "vmp"
        assert l1.profile_dir == "down"
        assert l1.time_reference_year == 2025

    def test_time_arrays(self):
        pf = MockPFile(n_slow=100)
        l1 = pfile_to_l1data(pf, speed=0.6)

        ratio = round(pf.fs_fast / pf.fs_slow)
        assert len(l1.time) == 100 * ratio
        assert len(l1.time_slow) == 100

    def test_pressure_interpolated(self):
        pf = MockPFile(n_slow=200, pressure_ramp=(10.0, 30.0))
        l1 = pfile_to_l1data(pf, speed=0.5)

        # Endpoints of pressure ramp should be preserved after interp
        np.testing.assert_allclose(l1.pres[0], 10.0, atol=0.5)
        np.testing.assert_allclose(l1.pres[-1], 30.0, atol=0.5)
        assert len(l1.pres) == len(l1.time)

    def test_temperature_interpolated(self):
        pf = MockPFile(n_slow=200)
        l1 = pfile_to_l1data(pf, speed=0.5)

        assert len(l1.temp) == len(l1.time)
        # Should be interpolated from ~20 to ~5
        assert l1.temp[0] > l1.temp[-1]

    def test_shear_shape(self):
        pf = MockPFile(shear_names=("sh1", "sh2", "sh3"))
        l1 = pfile_to_l1data(pf, speed=0.6)

        assert l1.shear.shape == (3, len(l1.time))

    def test_no_shear_channels(self):
        pf = MockPFile(shear_names=())
        l1 = pfile_to_l1data(pf, speed=0.5)

        assert l1.n_shear == 0
        assert l1.shear.shape[0] == 0

    def test_no_pressure_raises(self):
        pf = MockPFile()
        del pf.channels["P"]
        with pytest.raises(ValueError, match="No pressure channel"):
            pfile_to_l1data(pf, speed=0.5)

    def test_P_dP_preferred_over_P(self):
        """P_dP channel should be preferred when present."""
        pf = MockPFile(n_slow=100, pressure_ramp=(10.0, 20.0))
        pf.channels["P_dP"] = np.linspace(100.0, 200.0, 100)
        pf.channel_info["P_dP"] = {"type": "pres", "units": "dbar"}
        l1 = pfile_to_l1data(pf, speed=0.5)

        # Should use P_dP (100-200 range), not P (10-20 range)
        assert l1.pres[0] > 50.0


class TestSpeedComputation:
    """Verify dP/dt-based speed computation and fixed speed override."""

    def test_fixed_speed(self):
        pf = MockPFile(n_slow=200)
        l1 = pfile_to_l1data(pf, speed=0.7)

        np.testing.assert_allclose(l1.pspd_rel, 0.7)

    def test_fixed_speed_positive(self):
        """Fixed speed should always be positive regardless of sign input."""
        pf = MockPFile(n_slow=200)
        l1 = pfile_to_l1data(pf, speed=-0.5)

        np.testing.assert_allclose(l1.pspd_rel, 0.5)

    def test_computed_speed_positive(self):
        """Speed from dP/dt should be positive (absolute value)."""
        pf = MockPFile(n_slow=500, pressure_ramp=(5.0, 50.0))
        l1 = pfile_to_l1data(pf)

        assert np.all(l1.pspd_rel > 0)

    def test_computed_speed_minimum_floor(self):
        """Computed speed should be clipped to >= 0.05 m/s."""
        pf = MockPFile(n_slow=500, pressure_ramp=(10.0, 10.001))
        l1 = pfile_to_l1data(pf)

        assert np.all(l1.pspd_rel >= 0.05)


class TestDirectionHandling:
    """Direction parameter should propagate to L1Data."""

    def test_direction_down(self):
        pf = MockPFile()
        l1 = pfile_to_l1data(pf, speed=0.5, direction="down")
        assert l1.profile_dir == "down"

    def test_direction_up(self):
        pf = MockPFile()
        l1 = pfile_to_l1data(pf, speed=0.5, direction="up")
        assert l1.profile_dir == "up"


class TestProfileSlicing:
    """Verify fast/slow index mapping when profile_slice is provided."""

    def test_sliced_lengths(self):
        n_slow = 500
        pf = MockPFile(n_slow=n_slow)
        ratio = round(pf.fs_fast / pf.fs_slow)

        s, e = 100, 299  # 200 slow samples
        l1 = pfile_to_l1data(pf, profile_slice=(s, e), speed=0.5)

        expected_slow = e - s + 1  # inclusive
        expected_fast = expected_slow * ratio
        assert len(l1.time_slow) == expected_slow
        assert len(l1.time) == expected_fast
        assert len(l1.pres) == expected_fast
        assert l1.shear.shape[1] == expected_fast

    def test_sliced_pressure_values(self):
        """Sliced pressure should correspond to the selected range."""
        pf = MockPFile(n_slow=400, pressure_ramp=(0.0, 40.0))
        # Each slow sample is 1 dbar apart (40/400 = 0.1 dbar/sample)
        s, e = 100, 199
        l1 = pfile_to_l1data(pf, profile_slice=(s, e), speed=0.5)

        expected_p_start = pf.channels["P"][s]
        expected_p_end = pf.channels["P"][e]
        np.testing.assert_allclose(l1.pres_slow[0], expected_p_start)
        np.testing.assert_allclose(l1.pres_slow[-1], expected_p_end)

    def test_full_file_when_no_slice(self):
        n_slow = 200
        pf = MockPFile(n_slow=n_slow)
        ratio = round(pf.fs_fast / pf.fs_slow)
        l1 = pfile_to_l1data(pf, speed=0.5)

        assert len(l1.time) == n_slow * ratio
        assert len(l1.time_slow) == n_slow


class TestVibrationChannels:
    """Accelerometers vs piezo vibration sensors."""

    def test_acc_preferred(self):
        pf = MockPFile(acc_names=("Ax", "Az"))
        l1 = pfile_to_l1data(pf, speed=0.5)

        assert l1.vib_type == "ACC"
        assert l1.n_vib == 2

    def test_piezo_fallback(self):
        pf = MockPFile(acc_names=(), piezo_names=("V1", "V2"))
        l1 = pfile_to_l1data(pf, speed=0.5)

        assert l1.vib_type == "VIB"
        assert l1.n_vib == 2

    def test_no_vibration(self):
        pf = MockPFile(acc_names=(), piezo_names=())
        l1 = pfile_to_l1data(pf, speed=0.5)

        assert l1.vib_type == "NONE"
        assert l1.n_vib == 0


class TestFastTemperature:
    """Fast thermistor channels (T1_dT1) for chi computation."""

    def test_temp_fast_from_dT_channels(self):
        pf = MockPFile(temp_fast_names=("T1_dT1",))
        l1 = pfile_to_l1data(pf, speed=0.5)

        assert l1.has_temp_fast
        assert l1.n_temp == 1
        assert l1.temp_fast.shape[1] == len(l1.time)

    def test_diff_gains(self):
        pf = MockPFile(temp_fast_names=("T1_dT1", "T2_dT2"))
        l1 = pfile_to_l1data(pf, speed=0.5)

        assert len(l1.diff_gains) == 2
        assert all(g == 0.94 for g in l1.diff_gains)

    def test_no_temp_fast(self):
        pf = MockPFile(temp_fast_names=(), has_T1=False)
        l1 = pfile_to_l1data(pf, speed=0.5)

        assert not l1.has_temp_fast

    def test_T_fast_fallback(self):
        """When no T*_dT* channels exist, fast T channels are used."""
        pf = MockPFile(temp_fast_names=())
        # Add T1 as a fast channel to trigger fallback
        pf._fast_channels.add("T1")
        l1 = pfile_to_l1data(pf, speed=0.5)

        # T1 is in _fast_channels and matches T_re, so should be picked up
        assert l1.has_temp_fast
        assert l1.n_temp == 1


class TestL1DataAttributes:
    """Verify derived L1Data properties are correct after conversion."""

    def test_n_shear_n_vib(self):
        pf = MockPFile(shear_names=("sh1",), acc_names=("Ax", "Ay", "Az"))
        l1 = pfile_to_l1data(pf, speed=0.5)

        assert l1.n_shear == 1
        assert l1.n_vib == 3

    def test_has_speed(self):
        pf = MockPFile()
        l1 = pfile_to_l1data(pf, speed=0.5)
        assert l1.has_speed

    def test_n_time(self):
        pf = MockPFile(n_slow=300)
        ratio = round(pf.fs_fast / pf.fs_slow)
        l1 = pfile_to_l1data(pf, speed=0.5)
        assert l1.n_time == 300 * ratio
