# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for .p to ATOMIX L1_converted NetCDF conversion."""

from pathlib import Path

import numpy as np
import pytest

TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_FILE = TEST_DATA_DIR / "SN479_0006.p"


@pytest.fixture
def skip_no_data():
    if not SAMPLE_FILE.exists():
        pytest.skip("Test data not available")


def test_p_to_L1_roundtrip(skip_no_data, tmp_path):
    """Convert .p -> L1 .nc, read back, verify ATOMIX group structure."""
    import netCDF4 as nc

    from odas_tpw.rsi.convert import p_to_L1

    out_path = tmp_path / "test_output.nc"
    pf, nc_path = p_to_L1(SAMPLE_FILE, out_path)

    assert nc_path.exists()
    assert nc_path.stat().st_size > 0

    ds = nc.Dataset(str(nc_path), "r")

    # Check root dimensions
    assert "TIME" in ds.dimensions
    assert "TIME_SLOW" in ds.dimensions
    assert len(ds.dimensions["TIME"]) == len(pf.t_fast)
    assert len(ds.dimensions["TIME_SLOW"]) == len(pf.t_slow)

    # Check L1_converted group exists
    assert "L1_converted" in ds.groups
    L1 = ds.groups["L1_converted"]

    # Check group attributes
    assert float(L1.fs_fast) == pytest.approx(pf.fs_fast, rel=1e-3)
    assert float(L1.fs_slow) == pytest.approx(pf.fs_slow, rel=1e-3)
    assert hasattr(L1, "vehicle")
    assert hasattr(L1, "f_AA")
    assert hasattr(L1, "time_reference_year")

    # Check core ATOMIX variables exist in L1 group
    assert "TIME" in L1.variables
    assert "TIME_SLOW" in L1.variables
    assert "PRES" in L1.variables
    assert "PRES_SLOW" in L1.variables

    # Check pressure values roundtrip
    P_slow_nc = L1.variables["PRES_SLOW"][:].data
    P_src = pf.channels.get("P_dP", pf.channels.get("P"))
    np.testing.assert_allclose(P_slow_nc, P_src, rtol=1e-10)

    ds.close()


def test_p_to_L1_default_path(skip_no_data, tmp_path, monkeypatch):
    """p_to_L1 with nc_filepath=None should write next to the .p file."""
    import shutil

    local_p = tmp_path / SAMPLE_FILE.name
    shutil.copy2(SAMPLE_FILE, local_p)

    from odas_tpw.rsi.convert import p_to_L1

    _pf, nc_path = p_to_L1(local_p)
    assert nc_path.exists()
    assert nc_path.suffix == ".nc"
    assert nc_path.parent == tmp_path


def test_p_to_L1_shear_sensors(skip_no_data, tmp_path):
    """SHEAR variable should have sensor-indexed dimension."""
    import netCDF4 as nc

    from odas_tpw.rsi.convert import p_to_L1

    out_path = tmp_path / "test_shear.nc"
    pf, nc_path = p_to_L1(SAMPLE_FILE, out_path)

    ds = nc.Dataset(str(nc_path), "r")
    L1 = ds.groups["L1_converted"]

    # Count shear channels in source
    n_shear = sum(1 for n in pf.channels if pf.channel_info[n]["type"] == "shear")
    if n_shear > 0:
        assert "N_SHEAR_SENSORS" in ds.dimensions
        assert len(ds.dimensions["N_SHEAR_SENSORS"]) == n_shear
        assert "SHEAR" in L1.variables
        shear = L1.variables["SHEAR"]
        assert shear.shape == (n_shear, len(pf.t_fast))
        assert shear.units == "s-1"
        assert shear.standard_name == "sea_water_velocity_shear"

    ds.close()


def test_p_to_L1_gradt(skip_no_data, tmp_path):
    """GRADT should contain temperature gradient in K/m."""
    import netCDF4 as nc

    from odas_tpw.rsi.convert import p_to_L1

    out_path = tmp_path / "test_gradt.nc"
    pf, nc_path = p_to_L1(SAMPLE_FILE, out_path)

    ds = nc.Dataset(str(nc_path), "r")
    L1 = ds.groups["L1_converted"]

    n_gradt = sum(1 for n in pf.channels if pf.channel_info[n]["type"] == "therm" and pf.is_fast(n))
    if n_gradt > 0:
        assert "GRADT" in L1.variables
        gradt = L1.variables["GRADT"]
        assert gradt.units == "degrees_Celsius m-1"
        # Should be finite (no NaN/Inf in the bulk)
        data = gradt[:].data
        assert np.isfinite(data).sum() > 0.9 * data.size

    ds.close()


def test_convert_all_serial(skip_no_data, tmp_path):
    """convert_all with jobs=1 (serial) should convert files."""
    from odas_tpw.rsi.convert import convert_all

    convert_all([SAMPLE_FILE], tmp_path, jobs=1)
    nc_files = list(tmp_path.glob("*.nc"))
    assert len(nc_files) == 1
    assert nc_files[0].stat().st_size > 0


def test_convert_all_parallel(skip_no_data, tmp_path):
    """convert_all with jobs=2 (parallel) should convert files."""
    from odas_tpw.rsi.convert import convert_all

    convert_all([SAMPLE_FILE], tmp_path, jobs=2)
    nc_files = list(tmp_path.glob("*.nc"))
    assert len(nc_files) == 1
    assert nc_files[0].stat().st_size > 0


def test_cf_compliance(skip_no_data, tmp_path):
    """Output NetCDF should be CF-1.13 compliant with ATOMIX structure."""
    import netCDF4 as nc

    from odas_tpw.rsi.convert import p_to_L1

    out_path = tmp_path / "test_cf.nc"
    _pf, nc_path = p_to_L1(SAMPLE_FILE, out_path)
    ds = nc.Dataset(str(nc_path), "r")

    # Required global attributes
    assert ds.Conventions == "CF-1.13, ACDD-1.3"
    assert hasattr(ds, "title")
    assert hasattr(ds, "history")

    L1 = ds.groups["L1_converted"]

    # Time coordinate variables in L1 group
    for tvar_name in ("TIME", "TIME_SLOW"):
        tvar = L1.variables[tvar_name]
        assert tvar.standard_name == "time"
        assert tvar.axis == "T"
        assert "Days since" in tvar.units

    # Pressure variable CF attributes
    P_var = L1.variables["PRES"]
    assert P_var.standard_name == "sea_water_pressure"

    ds.close()


def test_supplementary_units_canonicalized(skip_no_data, tmp_path):
    """Supplementary channels must carry UDUNITS-canonical units.

    Regression: convert.py:541 wrote RSI's raw unit strings verbatim (e.g.
    'deg_C', 'deg', 'mS_cm-1'), bypassing canonicalize_units. The SN479
    config exposes Incl_T with raw units 'deg_C', which must be normalized
    to 'degree_Celsius'.
    """
    import netCDF4 as nc

    from odas_tpw.perturb.netcdf_schema import canonicalize_units
    from odas_tpw.rsi.convert import _classify_channels, p_to_L1
    from odas_tpw.rsi.p_file import PFile

    pf = PFile(SAMPLE_FILE)
    supplementary = _classify_channels(pf)["supplementary"]
    # At least one supplementary channel should need canonicalization
    raw_units = {n: pf.channel_info[n]["units"] for n in supplementary}
    needs_fix = {n: u for n, u in raw_units.items() if canonicalize_units(u) != u}
    assert needs_fix, f"test precondition: no canonicalizable units in {raw_units}"

    out_path = tmp_path / "test_supp_units.nc"
    p_to_L1(SAMPLE_FILE, out_path)

    ds = nc.Dataset(str(out_path), "r")
    L1 = ds.groups["L1_converted"]
    for name in needs_fix:
        var = L1.variables[name.replace(" ", "_")]
        # Must be the canonical form, never the raw RSI string
        assert var.units == canonicalize_units(raw_units[name])
        assert var.units != raw_units[name]
    ds.close()


def test_mag_units_udunits_parseable():
    """MAG spec must declare a UDUNITS-parseable unit, not 'micro_Tesla'.

    Regression: convert.py:361 hardcoded units='micro_Tesla', which UDUNITS
    rejects (UT_UNKNOWN). No magnetometer in the SN479 config, so inspect
    the spec directly with a synthetic single-MAG channel.
    """
    import types

    import numpy as np

    from odas_tpw.rsi.convert import _l1_variable_specs

    n = 4
    pf = types.SimpleNamespace(channels={"Mx": np.zeros(n)}, is_fast=lambda _name: False)
    ch = {
        "shear": [],
        "vib": [],
        "acc": [],
        "mag": ["Mx"],
        "gradt": [],
        "cond_ctd": [],
        "temp_ctd": None,
        "pitch": None,
        "roll": None,
        "chla": None,
        "turb": None,
        "doxy": None,
        "doxy_temp": None,
    }
    z = np.zeros(n)
    specs = _l1_variable_specs(pf, ch, z, z, "Days since 2025-01-01T00:00:00Z", z, z, z, None)
    mag_attrs = next(attrs for name, _dt, _dim, _data, attrs in specs if name == "MAG")
    assert mag_attrs["units"] == "uT"

    cf_units = pytest.importorskip("cf_units")
    cf_units.Unit(mag_attrs["units"])  # raises if not UDUNITS-parseable


def test_convert_all_output_name_collision(tmp_path):
    """Two same-named .p files in different dirs must not silently collide."""
    from odas_tpw.rsi.convert import convert_all

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    f1 = a / "SN479_0001.p"
    f2 = b / "SN479_0001.p"
    f1.write_bytes(b"")
    f2.write_bytes(b"")

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="collision"):
        convert_all([f1, f2], out, jobs=1)


def test_convert_all_skips_corrupt_file(tmp_path, caplog):
    """A truncated .p file must be logged, not abort the batch (serial)."""
    import logging

    from odas_tpw.rsi.convert import convert_all

    tiny = tmp_path / "tiny.p"
    tiny.write_bytes(b"\x00\x01\x02")  # < 128-byte header

    out = tmp_path / "out"
    # Must not raise; conversion fails per-file and is logged
    with caplog.at_level(logging.ERROR):
        convert_all([tiny], out, jobs=1)
    assert any("tiny.p" in rec.message for rec in caplog.records)
    # No output produced for the corrupt file
    assert not list(out.glob("*.nc"))


def test_convert_all_catches_struct_error(tmp_path, caplog, monkeypatch):
    """convert_all must catch struct.error per-file, not abort the batch.

    Regression: the log-and-continue except clauses listed only
    (OSError, ValueError, RuntimeError); a struct.error raised by PFile on a
    malformed .p file would escape and abort the whole batch. PFile now
    raises ValueError for short headers, but the defensive broadening here is
    validated by forcing the worker to raise struct.error directly.
    """
    import logging
    import struct

    import odas_tpw.rsi.convert as convert_mod
    from odas_tpw.rsi.convert import convert_all

    bad = tmp_path / "bad.p"
    bad.write_bytes(b"\x00" * 8)

    def _boom(_args):
        raise struct.error("unpack_from requires a larger buffer")

    monkeypatch.setattr(convert_mod, "_convert_one", _boom)

    out = tmp_path / "out"
    # Must not raise struct.error; it is caught and logged.
    with caplog.at_level(logging.ERROR):
        convert_all([bad], out, jobs=1)
    assert any("bad.p" in rec.message for rec in caplog.records)


def test_gradt_single_sample_no_crash(skip_no_data, tmp_path, monkeypatch):
    """GRADT must not IndexError on a one-sample fast channel.

    Regression (convert.py:514-515): np.diff on a single sample is empty, so
    dTdt[-1] raised IndexError. Drive the real p_to_L1 pre-compute loop by
    forcing every fast/slow channel and time vector down to a single sample.
    On OLD code this raises IndexError; with the size<2 guard it succeeds.
    """
    import numpy as np

    import odas_tpw.rsi.convert as convert_mod
    import odas_tpw.rsi.p_file as p_file_mod
    from odas_tpw.rsi.convert import _classify_channels, p_to_L1
    from odas_tpw.rsi.p_file import PFile

    real_pfile = PFile

    class OneScanPFile(PFile):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Collapse every channel and time vector to a single sample so the
            # GRADT diff/append path runs on size-1 fast thermistor channels.
            self.channels = {n: v[:1].copy() for n, v in self.channels.items()}
            self.t_fast = self.t_fast[:1].copy()
            self.t_slow = self.t_slow[:1].copy()

    # p_to_L1 imports PFile locally from odas_tpw.rsi.p_file, so patch it there.
    monkeypatch.setattr(p_file_mod, "PFile", OneScanPFile)
    # Speed computation needs >=1 sample; stub it to a finite scalar array.
    monkeypatch.setattr(
        convert_mod,
        "_compute_speed",
        lambda pf: (np.array([0.5]), np.array([0.5])),
    )

    pf_probe = real_pfile(SAMPLE_FILE)
    if not _classify_channels(pf_probe)["gradt"]:
        pytest.skip("no fast thermistor channel in sample")

    out_path = tmp_path / "test_gradt_1scan.nc"
    # Must not raise IndexError
    _pf, nc_path = p_to_L1(SAMPLE_FILE, out_path)
    assert nc_path.exists()


def test_backward_compat_alias(skip_no_data, tmp_path):
    """p_to_netcdf should still work as an alias for p_to_L1."""
    from odas_tpw.rsi.convert import p_to_L1, p_to_netcdf

    assert p_to_netcdf is p_to_L1

    out_path = tmp_path / "test_alias.nc"
    _pf, nc_path = p_to_netcdf(SAMPLE_FILE, out_path)
    assert nc_path.exists()
