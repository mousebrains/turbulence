# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Profiles cut from an ``rsi-tpw nc`` (ATOMIX L1) file (issue #235).

``rsi-tpw prof`` on an L1 file used to copy only its 1-D variables, under
their ATOMIX names: every profile lacked the shear, vibration and FP07
channels and ``P``, and ``rsi-tpw eps`` on one died with ``KeyError: 'P'``.
The profiles now carry the channels a profile cut from the ``.p`` carries,
under the same names, with the attributes eps/chi read and the same RDL
bad-buffer masks, so eps and chi agree with the ``.p`` route bit for bit on
the committed VMP (SN479) and MicroRider-on-glider (MR_SL685_climb) files.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import netCDF4
import numpy as np
import pytest

from odas_tpw.rsi import config_patch as cp
from odas_tpw.rsi.bad_buffer import (
    L1_SPANS_ATTR,
    UNKNOWN_ATTR,
    decode_channel_spans,
    encode_channel_spans,
)
from odas_tpw.rsi.chi_io import compute_chi_file
from odas_tpw.rsi.convert import (
    _classify_channels,
    _pfile_layout,
    _single_channel_sources,
    l1_native_channels,
    p_to_L1,
)
from odas_tpw.rsi.dissipation import compute_diss_file
from odas_tpw.rsi.helpers import load_channels
from odas_tpw.rsi.p_file import SHEAR_INTERMEDIATE_COMMENT, PFile, parse_config
from odas_tpw.rsi.profile import _load_from_nc, extract_profiles
from tests.test_bad_buffer_masks import _inject_fast_run, _inject_slow_run

DATA = Path(__file__).parent / "data"
SN479 = DATA / "SN479_0006.p"
MR685 = DATA / "MR_SL685_climb.p"  # a MicroRider on a glider: vehicle, direction, tau
pytestmark = pytest.mark.skipif(
    not (SN479.exists() and MR685.exists()), reason="SN479/MR685 test data not available"
)

# Attributes that record WHEN and BY WHICH CODE a file was written, or what
# file it was read from: they differ between any two runs.
_RUN_ATTRS = {"software_version", "history", "source"}


class Case(NamedTuple):
    p: Path
    pf: PFile
    l1: Path
    from_p: list[Path]
    from_nc: list[Path]


def _build_case(p: Path, d: Path) -> Case:
    """``.p -> prof`` and ``.p -> nc -> prof``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = PFile(p)
        _, l1 = p_to_L1(p, d / f"{p.stem}.nc")
        from_p = extract_profiles(p, d / "from_p")
        from_nc = extract_profiles(l1, d / "from_nc")
    assert len(from_p) == len(from_nc) >= 1
    return Case(p, pf, l1, from_p, from_nc)


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    built: dict[Path, Case] = {}

    def get(p: Path) -> Case:
        if p not in built:
            built[p] = _build_case(p, tmp_path_factory.mktemp(p.stem))
        return built[p]

    return get


@pytest.fixture(scope="module", params=[SN479, MR685], ids=lambda p: p.stem)
def case(request, cases) -> Case:
    return cases(request.param)


@pytest.fixture(scope="module")
def sn479(cases) -> Case:
    return cases(SN479)


def _attrs(obj) -> dict:
    return {a: obj.getncattr(a) for a in obj.ncattrs()}


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))  # a NaN _FillValue
    return bool(np.array_equal(a, b))


def _values(var) -> np.ndarray:
    """A numeric variable as float64, masked (fill) values as NaN."""
    return np.asarray(np.ma.filled(var[:].astype(np.float64), np.nan))


def _bits(var) -> bytes | list:
    """A variable's stored values: the bytes of a numeric one, else the values."""
    data = np.ma.getdata(var[:])
    return data.tobytes() if data.dtype.kind in "fiub" else data.tolist()


def _assert_same_variables(a: Path, b: Path, *, units_differ=("Ax", "Ay")) -> None:
    """Same variables in the same order, bit for bit, with the same attributes
    (but for the units the L1 file states for the piezos)."""
    with netCDF4.Dataset(str(a)) as da, netCDF4.Dataset(str(b)) as db:
        assert list(db.variables) == list(da.variables), (a.name, b.name)
        for name, va in da.variables.items():
            vb = db.variables[name]
            assert vb.dimensions == va.dimensions, name
            assert vb.dtype == va.dtype, name
            assert _bits(vb) == _bits(va), (a.name, name)
            aa, ab = _attrs(va), _attrs(vb)
            if name in units_differ and "sensor_type" in aa:
                # The L1 file calls the piezos' units "1", PFile "counts".
                assert (aa.pop("units"), ab.pop("units")) == ("counts", "1")
            assert ab.keys() == aa.keys(), (a.name, name)
            assert all(_same(ab[k], aa[k]) for k in aa), (a.name, name)


def _products(prof: Path, out: Path) -> tuple[Path, Path]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_diss_file(prof, out / "eps")[0], compute_chi_file(prof, out / "chi")[0]


def _copy_l1(l1: Path, dst: Path, edit) -> Path:
    """A copy of *l1* edited in place by ``edit(root_dataset)``."""
    shutil.copy(l1, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        edit(ds)
    return dst


# ---------------------------------------------------------------------------
# The round trip on real files
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_same_channels_same_bits(self, case):
        for a, b in zip(case.from_p, case.from_nc, strict=True):
            _assert_same_variables(a, b)

    def test_global_attrs(self, case):
        """Everything the .p route writes, framing and stime/etime included;
        the L1 file's own ``source`` and its history line come along."""
        for a, b in zip(case.from_p, case.from_nc, strict=True):
            with netCDF4.Dataset(str(a)) as dp, netCDF4.Dataset(str(b)) as dn:
                gp, gn = _attrs(dp), _attrs(dn)
                for k in gp.keys() - _RUN_ATTRS:
                    assert _same(gn[k], gp[k]), k
                assert gn.keys() - gp.keys() == {"source"}
                assert gn["source"] == case.pf.config["instrument_info"]["model"]
                assert f"L1 converted from {case.p.name}" in gn["history"]
                assert "samples_per_record_fast" in gp and "record_offset_fast" in gp

    def test_epsilon_and_chi_are_bit_identical(self, case, tmp_path):
        for a, b in zip(
            _products(case.from_p[0], tmp_path / "p"),
            _products(case.from_nc[0], tmp_path / "nc"),
            strict=True,
        ):
            _assert_same_variables(a, b, units_differ=())
            with netCDF4.Dataset(str(a)) as dp, netCDF4.Dataset(str(b)) as dn:
                gp, gn = _attrs(dp), _attrs(dn)
                assert {k: str(v) for k, v in gn.items() if k not in _RUN_ATTRS} == {
                    k: str(v) for k, v in gp.items() if k not in _RUN_ATTRS
                }

    def test_vibration_stack_matches_the_p_route(self, case):
        dp, dn = load_channels(case.from_p[0]), load_channels(case.from_nc[0])
        assert [n for n, _ in dn["accel"]] == [n for n, _ in dp["accel"]] == ["Ax", "Ay"]
        assert dn["channel_types"] == dp["channel_types"] == {"Ax": "piezo", "Ay": "piezo"}


class TestInverse:
    """:func:`l1_native_channels` against the PFile the L1 file came from."""

    @pytest.fixture(scope="class")
    def restored(self, case):
        pf = case.pf
        with netCDF4.Dataset(str(case.l1)) as ds:
            config = parse_config(ds.configuration_string)
            return case, l1_native_channels(
                ds.groups["L1_converted"],
                config,
                pf.t_fast,
                pf.t_slow,
                pf.fs_fast,
                pf.fs_slow,
                software_version=ds.software_version,
            )

    def test_every_channel_with_its_rate_and_type(self, restored):
        case, channels = restored
        pf = case.pf
        assert list(channels) == list(pf.channels)
        for name, (_data, fast, info) in channels.items():
            assert fast == pf.is_fast(name), name
            assert info["type"] == pf.channel_info[name]["type"], name

    def test_values_against_pfile(self, restored):
        """Shear against PFile's own sh1/sh2: the speed normalization undone."""
        case, channels = restored
        pf = case.pf
        supplementary = _classify_channels(pf)["supplementary"]
        assert {"sh1", "sh2"} <= set(channels)
        for name, (data, _fast, _info) in channels.items():
            if pf.channel_info[name]["type"] == "shear":
                # x / s**2 * s**2: within an ulp, and identical once in float32.
                np.testing.assert_allclose(data, pf.channels[name], rtol=4e-16, atol=0)
                assert _same(data.astype(np.float32), pf.channels[name].astype(np.float32))
            elif name in supplementary:
                # p_to_L1 stores these as float32, as every profile file does.
                assert _same(data, pf.channels[name].astype(np.float32)), name
            else:
                assert _same(data, pf.channels[name]), name

    def test_shear_carries_the_intermediate_comment(self, restored):
        _, channels = restored
        assert channels["sh1"][2]["comment"] == SHEAR_INTERMEDIATE_COMMENT

    def test_single_channel_variables_hold_their_roles_channels(self, case):
        """The writer's tables, read forward: what the inverse relies on."""
        single = _single_channel_sources(_classify_channels(case.pf))
        with netCDF4.Dataset(str(case.l1)) as ds:
            g = ds.groups["L1_converted"]
            for var_name, ch_name in single.items():
                assert _same(g.variables[var_name][:], case.pf.channels[ch_name]), var_name

    def test_layout_is_the_pfiles(self, case):
        held, pairs = _pfile_layout(case.pf.config)
        pf = case.pf
        assert list(held.items()) == [(n, pf.channel_info[n]["type"]) for n in pf.channels]
        assert pairs == {"T1_dT1": "T1", "T2_dT2": "T2", "P_dP": "P"}


def test_sn479_single_channel_roles(sn479):
    assert _single_channel_sources(_classify_channels(sn479.pf)) == {
        "TEMP_CTD": "JAC_T",
        "COND_CTD": "JAC_C",
        "PITCH": "Incl_X",
        "ROLL": "Incl_Y",
        "CHLA": "Chlorophyll",
        "TURB": "Turbidity",
        "DOXY": "DO",
        "DOXY_TEMP": "DO_T",
    }


class TestTempLabel:
    """TEMP's ``sensor_names`` reads ``T1_dT1, T2_dT2``. That is not a mislabel:
    ``sensor_names`` names the channel each row is converted from, and PFile
    keeps the deconvolved fast temperature under the pre-emphasized channel's
    name. So the label stays, and the reader takes it as written."""

    def test_temp_rows_are_the_channels_they_name(self, sn479):
        with netCDF4.Dataset(str(sn479.l1)) as ds:
            temp = ds.groups["L1_converted"].variables["TEMP"]
            names = [s.strip() for s in temp.sensor_names.split(",")]
            assert names == ["T1_dT1", "T2_dT2"]
            for name, row in zip(names, temp[:], strict=True):
                assert sn479.pf.channel_info[name]["units"] == "deg_C"
                assert _same(row, sn479.pf.channels[name]), name

    def test_a_plain_fast_thermistor_label(self, tmp_path):
        """An instrument without pre-emphasis files its fast thermistor itself
        (``T1``): restored fast, with no slow base invented, and with the plain-T
        attributes chi's first-difference route reads."""
        from odas_tpw.rsi.chi_io import _DEFAULT_DIFF_GAIN
        from odas_tpw.rsi.profile import _channel_attrs

        config, path = _synthetic_l1(tmp_path)
        restored = _restore(config, path)
        data, fast, info = restored["T1"]
        assert fast and info["type"] == "therm"
        assert _same(data, np.full(4 * _N_SLOW, 15.0))  # the TEMP row
        assert "T1_dT1" not in restored
        attrs = _channel_attrs("T1", "time_fast", info, config)
        assert attrs["diff_gain"] == _DEFAULT_DIFF_GAIN
        assert attrs["T_0"] == pytest.approx(289.3)


# ---------------------------------------------------------------------------
# RDL bad-buffer masks travel through the L1 file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dropouts(sn479, tmp_path_factory) -> Case:
    """SN479 with RDL runs injected inside its profile: 256 samples of sh1
    (0.5 s, dropped), 64 of the slow T1 (1 s, dropped; a channel the L1 file
    itself leaves out) and 8 of JAC_T (0.125 s, interpolated)."""
    with netCDF4.Dataset(str(sn479.from_p[0])) as ds:
        cycle = (int(ds.profile_start_index_slow) + int(ds.profile_end_index_slow)) // 2
    raw = SN479.read_bytes()
    raw = _inject_fast_run(raw, sn479.pf, "sh1", cycle, 256)
    raw = _inject_slow_run(raw, sn479.pf, "T1", cycle + 400, 64)
    raw = _inject_slow_run(raw, sn479.pf, "JAC_T", cycle - 400, 8)
    d = tmp_path_factory.mktemp("dropouts")
    p = d / "SN479_dropouts_0006.p"
    p.write_bytes(raw)
    return _build_case(p, d)


class TestBadBuffers:
    def test_the_writer_records_every_channels_spans(self, sn479, dropouts):
        with netCDF4.Dataset(str(sn479.l1)) as ds:
            assert ds.groups["L1_converted"].getncattr(L1_SPANS_ATTR) == ""  # clean, and says so
        with netCDF4.Dataset(str(dropouts.l1)) as ds:
            spans = decode_channel_spans(ds.groups["L1_converted"].getncattr(L1_SPANS_ATTR))
        assert set(spans) == {"sh1", "T1", "JAC_T"}

    def test_profiles_carry_the_p_routes_spans(self, dropouts):
        a, b = dropouts.from_p[0], dropouts.from_nc[0]
        with netCDF4.Dataset(str(a)) as ds:
            spanned = {v for v in ds.variables if "bad_buffer_spans" in ds.variables[v].ncattrs()}
        assert spanned == {"sh1", "T1", "JAC_T"}
        _assert_same_variables(a, b)  # bad_buffer_spans included

    def test_epsilon_and_chi_mask_identically(self, dropouts, tmp_path):
        (ep, cp_), (en, cn) = (
            _products(dropouts.from_p[0], tmp_path / "p"),
            _products(dropouts.from_nc[0], tmp_path / "nc"),
        )
        for a, b in ((ep, en), (cp_, cn)):
            _assert_same_variables(a, b, units_differ=())
        with netCDF4.Dataset(str(en)) as ds:
            frac = _values(ds.variables["bad_buffer_fraction"])
            eps = _values(ds.variables["epsilon"])
            assert (frac > 0).any() and np.isnan(eps[frac > 0]).all()
        with netCDF4.Dataset(str(cn)) as ds:
            assert (_values(ds.variables["bad_buffer_fraction"]) > 0).any()  # T1, via T1_dT1

    def test_an_older_l1_file_reads_unknown_not_clean(self, dropouts, tmp_path, caplog):
        """Without the spans attribute the file cannot say: the fractions are
        NaN, not the 0 that would read as clean, and both steps warn."""
        path = _copy_l1(
            dropouts.l1,
            tmp_path / dropouts.l1.name,
            lambda ds: ds.groups["L1_converted"].delncattr(L1_SPANS_ATTR),
        )
        with caplog.at_level(logging.WARNING):
            prof = extract_profiles(path, tmp_path / "prof")[0]
        assert f"{path.name} was written by an rsi-tpw nc that did not record" in caplog.text
        with netCDF4.Dataset(str(prof)) as ds:
            assert UNKNOWN_ATTR in ds.ncattrs()
            assert not any("bad_buffer_spans" in v.ncattrs() for v in ds.variables.values())
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            eps, chi = _products(prof, tmp_path)
        assert f"{prof.name}: bad-buffer spans unknown" in caplog.text
        for product in (eps, chi):
            with netCDF4.Dataset(str(product)) as ds:
                assert UNKNOWN_ATTR in ds.ncattrs()
                for name in ("bad_buffer_fraction", "interpolated_fraction"):
                    assert np.isnan(_values(ds.variables[name])).all(), (product.name, name)

    def test_span_codec(self):
        masks = {"sh1": np.array([0, 2, 2, 0], np.int8), "a=b": np.array([1, 0], np.int8)}
        masks["clean"] = np.zeros(3, np.int8)
        text = encode_channel_spans(masks)
        assert text == "sh1=1:2:2;a=b=0:1:1"
        assert decode_channel_spans(text) == {"sh1": "1:2:2", "a=b": "0:1:1"}
        assert decode_channel_spans("") == {}


# ---------------------------------------------------------------------------
# A base channel the matrix does not sample (PFile builds it from X_dX)
# ---------------------------------------------------------------------------


def test_unsampled_base_is_rebuilt(sn479, tmp_path):
    """T1 configured but not in the matrix: PFile still builds it from T1_dT1,
    and the L1 route rebuilds it too."""
    text = cp.read_config_text(SN479)
    text, n = re.subn(r"(?m)^(id\s*=\s*)4(\s*)$", r"\g<1>99\g<2>", text)
    assert n == 1
    p = tmp_path / "SN479_noT1_0006.p"
    cp.write_patched_pfile(SN479, p, text)
    case = _build_case(p, tmp_path)
    assert "T1" in case.pf.channels and not case.pf.is_fast("T1")
    assert list(case.pf.channels)[-1] == "T1"  # held after the sampled channels
    held, pairs = _pfile_layout(case.pf.config)
    assert list(held)[-1] == "T1" and pairs["T1_dT1"] == "T1"
    for a, b in zip(case.from_p, case.from_nc, strict=True):
        _assert_same_variables(a, b)


def test_unconfigured_base_is_raw():
    text = _SYNTH_CONFIG.replace("name = T1\ntype = therm", "name = T3\ntype = therm")
    config = parse_config(text)
    config["channels"].append({"id": "3", "name": "T9_dT9", "type": "therm", "diff_gain": "1"})
    held, pairs = _pfile_layout(config)
    assert pairs["T9_dT9"] == "T9" and held["T9"] == "raw"
    assert held["T9_dT9"] == "therm"  # no T9 section to take a type from


# ---------------------------------------------------------------------------
# The mapping on a synthetic file: every rule in one place
# ---------------------------------------------------------------------------

_N_SLOW = 512
_FS_SLOW = 64.0
_SYNTH_GRID = (
    np.arange(4 * _N_SLOW) / (4 * _FS_SLOW),
    np.arange(_N_SLOW) / _FS_SLOW,
    4 * _FS_SLOW,
    _FS_SLOW,
)
_CURRENT = "microstructure-tpw 0.4.0 (git v0.4.0-43-g5d2b93cd8de3)"
# Four rows of four columns: sh1, Ax and T1 fast (a full column); P, JAC_T,
# V_Bat and P_dP slow. Incl_X is configured but not in the matrix.
_SYNTH_CONFIG = """
[matrix]
row01 = 1 2 3 10
row02 = 1 2 3 11
row03 = 1 2 3 12
row04 = 1 2 3 13
[instrument_info]
model = SYNTH
vehicle = vmp
[channel]
id = 1
name = sh1
type = shear
[channel]
id = 2
name = Ax
type = piezo
[channel]
id = 3
name = T1
type = therm
beta_1 = 3000
t_0 = 289.3
[channel]
id = 10
name = P
type = poly
[channel]
id = 11
name = JAC_T
type = jac_t
[channel]
id = 12
name = V_Bat
type = voltage
[channel]
id = 13
name = P_dP
type = poly
diff_gain = 20.5
[channel]
id = 20
name = Incl_X
type = inclxy
"""


def _synthetic_l1(tmp_path: Path, *, shear_names: str | None = "sh1", pres=None):
    """A hand-built L1 file in p_to_L1's layout and its parsed configuration."""
    n_slow, n_fast = _N_SLOW, 4 * _N_SLOW
    t = np.arange(n_fast)
    path = tmp_path / "synthetic_L1.nc"
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.configuration_string = _SYNTH_CONFIG
        ds.createDimension("TIME", n_fast)
        ds.createDimension("TIME_SLOW", n_slow)
        ds.createDimension("N_SHEAR_SENSORS", 1)
        ds.createDimension("N_VIB_SENSORS", 1)
        ds.createDimension("N_GRADT_SENSORS", 1)
        ds.createDimension("N_TEMP_SENSORS", 1)
        g = ds.createGroup("L1_converted")

        def var(name, dims, data, **attrs):
            v = g.createVariable(name, "f8", dims)
            v[:] = data
            for k, val in attrs.items():
                setattr(v, k, val)

        var("TIME", ("TIME",), 10.0 + t / (4 * _FS_SLOW) / 86400.0, units="Days since 2025-01-01")
        var("TIME_SLOW", ("TIME_SLOW",), 10.0 + np.arange(n_slow) / _FS_SLOW / 86400.0)
        p_slow = np.linspace(1.0, 50.0, n_slow) if pres is None else pres
        var("PRES_SLOW", ("TIME_SLOW",), p_slow, units="decibar")
        var("PRES", ("TIME",), np.linspace(1.0, 50.0, n_fast), units="decibar")
        shear = {"sensor_names": shear_names} if shear_names is not None else {}
        var("SHEAR", ("N_SHEAR_SENSORS", "TIME"), np.sin(t)[None, :], units="s-1", **shear)
        var("VIB", ("N_VIB_SENSORS", "TIME"), np.cos(t)[None, :], units="1", sensor_names="Ax")
        var("GRADT", ("N_GRADT_SENSORS", "TIME"), np.ones((1, n_fast)), sensor_names="T1")
        var("TEMP", ("N_TEMP_SENSORS", "TIME"), 15.0 + 0 * t[None, :], sensor_names="T1")
        var("TEMP_CTD", ("TIME_SLOW",), np.full(n_slow, 14.0), units="degree_Celsius")
        var("V_Bat", ("TIME_SLOW",), np.full(n_slow, 12.0), units="V", sensor_type="voltage")
        g.variables["V_Bat"].long_name = "V_Bat"
        var("FOO", ("TIME_SLOW",), np.zeros(n_slow), units="1")
    return parse_config(_SYNTH_CONFIG), path


def _restore(config, path, **kw):
    kw.setdefault("software_version", _CURRENT)
    with netCDF4.Dataset(str(path)) as ds:
        return l1_native_channels(ds.groups["L1_converted"], config, *_SYNTH_GRID, **kw)


class TestMapping:
    @pytest.fixture
    def synthetic(self, tmp_path):
        return _synthetic_l1(tmp_path)

    def test_names_order_rates_and_types(self, synthetic):
        restored = _restore(*synthetic)
        # Configuration order, then what the configuration does not name.
        assert list(restored) == ["sh1", "Ax", "T1", "P", "JAC_T", "V_Bat", "P_dP", "FOO"]
        rates = {n: fast for n, (_d, fast, _i) in restored.items()}
        assert [n for n, fast in rates.items() if fast] == ["sh1", "Ax", "T1"]
        types = {n: info["type"] for n, (_d, _f, info) in restored.items()}
        assert types == {
            "sh1": "shear",
            "Ax": "piezo",
            "T1": "therm",
            "P": "poly",
            "JAC_T": "jac_t",
            "V_Bat": "voltage",
            "P_dP": "poly",
            "FOO": "",
        }

    def test_derived_variables_are_not_channels(self, synthetic):
        restored = _restore(*synthetic)
        assert not {"GRADT", "PRES", "PRES_SLOW", "TEMP", "TEMP_CTD"} & set(restored)

    def test_slow_pre_emphasized_pair_repeats_the_base(self, synthetic):
        restored = _restore(*synthetic)
        assert _same(restored["P_dP"][0], restored["P"][0])

    def test_unsampled_channel_is_not_expected(self, synthetic, caplog):
        with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.convert"):
            restored = _restore(*synthetic)
        assert "Incl_X" not in restored
        assert caplog.text == ""

    def test_a_missing_configured_channel_is_named(self, tmp_path, caplog):
        config, path = _synthetic_l1(tmp_path)
        config["matrix"].append([1, 2, 3, 21])  # a slow channel the file never got
        config["channels"].append({"id": "21", "name": "JAC_C", "type": "jac_c"})
        with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.convert"):
            _restore(config, path, context="synthetic")
        assert "synthetic: configured channel(s) JAC_C are not in the L1 file" in caplog.text

    def test_without_sensor_names_rows_are_kept_as_stored(self, tmp_path, caplog):
        config, path = _synthetic_l1(tmp_path, shear_names=None)
        with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.convert"):
            restored = _restore(config, path)
        assert "SHEAR: sensor_names None does not name each of its 1 rows" in caplog.text
        data, fast, info = restored["SHEAR_1"]
        assert fast and info["type"] == "shear"
        assert "as stored" in info["comment"]
        assert _same(data, np.sin(np.arange(4 * _N_SLOW)))  # still speed-normalized
        assert "sh1" not in restored

    def test_without_configuration_single_variables_keep_l1_names(self, synthetic):
        _config, path = synthetic
        restored = _restore(None, path)
        assert "TEMP_CTD" in restored and "JAC_T" not in restored
        assert restored["P"][2]["type"] == ""
        assert "P_dP" not in restored  # no configuration, no pairs
        assert restored["Ax"][2]["type"] == "piezo"  # the role still types the row

    def test_a_name_used_twice_is_refused(self, tmp_path):
        config, path = _synthetic_l1(tmp_path)
        with netCDF4.Dataset(str(path), "a") as ds:
            ds.groups["L1_converted"].variables["VIB"].sensor_names = "sh1"
        with pytest.raises(ValueError, match="names channel 'sh1' a second time"):
            _restore(config, path)

    def test_a_nan_in_pressure_does_not_spread_through_the_speed(self, tmp_path):
        """Repaired as for profile detection, linearly: on a linear ramp the
        speed, and so the restored shear, is what it would have been."""
        clean = _restore(*_synthetic_l1(tmp_path))["sh1"][0]
        pres = np.linspace(1.0, 50.0, _N_SLOW)
        pres[200] = np.nan
        sh1 = _restore(*_synthetic_l1(tmp_path, pres=pres))["sh1"][0]
        assert np.isfinite(sh1).all()
        np.testing.assert_allclose(sh1, clean, rtol=1e-9)

    @pytest.mark.parametrize(
        ("version", "warned"),
        [
            (_CURRENT, False),
            ("microstructure-tpw 0.3.0", False),
            ("microstructure-tpw 0.2.9 (git v0.2.9)", True),
            (None, True),
            ("an unknown writer", True),
        ],
    )
    def test_shear_speed_provenance(self, synthetic, caplog, version, warned):
        with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.convert"):
            _restore(*synthetic, software_version=version)
        assert ("is not v0.3.0 or later" in caplog.text) is warned


# ---------------------------------------------------------------------------
# Edited copies of the real file
# ---------------------------------------------------------------------------


class TestEditedL1:
    def test_without_sensor_names(self, sn479, tmp_path, caplog):
        """Rows nobody named stay as stored under their L1 names; piezo rows
        are still typed, so they still join the Goodman stack."""

        def strip(ds):
            for v in ("SHEAR", "VIB", "TEMP", "GRADT"):
                ds.groups["L1_converted"].variables[v].delncattr("sensor_names")

        path = _copy_l1(sn479.l1, tmp_path / "SN479_0006.nc", strip)
        with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.convert"):
            prof = extract_profiles(path, tmp_path / "prof")[0]
        assert "kept as stored, as SHEAR_1..SHEAR_2" in caplog.text
        assert "sh1, sh2" in caplog.text  # named among the missing channels
        with netCDF4.Dataset(str(prof)) as ds:
            names = set(ds.variables)
            assert {"SHEAR_1", "SHEAR_2", "VIB_1", "VIB_2", "TEMP_1", "TEMP_2"} <= names
            assert not {"sh1", "Ax", "T1_dT1", "T1", "GRADT_1"} & names
        data = load_channels(prof)
        assert [n for n, _ in data["accel"]] == ["VIB_1", "VIB_2"]
        assert data["shear"] == []

    def test_without_configuration_string(self, sn479, tmp_path):
        """No configuration: stacked rows still restore by sensor_names and P
        from PRES_SLOW; the single-channel variables keep their L1 names; no FP07
        coefficients are made up."""
        path = _copy_l1(
            sn479.l1, tmp_path / "SN479_0006.nc", lambda ds: ds.delncattr("configuration_string")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prof = extract_profiles(path, tmp_path / "prof")[0]
        with netCDF4.Dataset(str(prof)) as ds:
            names = set(ds.variables)
            assert {"sh1", "sh2", "Ax", "Ay", "T1_dT1", "T2_dT2", "P"} <= names
            assert {"TEMP_CTD", "COND_CTD", "PITCH", "ROLL"} <= names
            assert not {"T1", "P_dP", "JAC_T"} & names
            assert "diff_gain" not in ds.variables["T1_dT1"].ncattrs()
            assert ds.variables["Ax"].sensor_type == "piezo"


# ---------------------------------------------------------------------------
# The vibration stack, on hardware where the type decides it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_p(tmp_path_factory) -> Path:
    """SN479 with a piezo named outside ^A[xyz]$ (Ay -> Vy) and a true
    accelerometer Az (the fast Turbidity column): only ``sensor_type`` puts Vy
    in the stack (the #168 fixture, rebuilt here)."""
    text = cp.read_config_text(SN479)
    for old, new in (("Ay", "Vy"), ("Turbidity", "Az")):
        text, n = re.subn(rf"(?m)^(name\s*=\s*){old}(\s*)$", rf"\g<1>{new}\g<2>", text)
        assert n == 1, old
    spec = cp.EditSpec(
        note="#235 synthetic mixed",
        author="test",
        channels={"Az": {"type": "accel", "coef0": "2.048", "coef1": "0.4"}},
    )
    text, changes = cp.edit_config_text(text, spec)
    assert changes
    dst = tmp_path_factory.mktemp("mixed") / "SN479_mixed_0006.p"
    cp.write_patched_pfile(SN479, dst, text)
    return dst


def test_vibration_stack_on_mixed_hardware(mixed_p, tmp_path):
    case = _build_case(mixed_p, tmp_path)
    dp, dn = load_channels(case.from_p[0]), load_channels(case.from_nc[0])
    stack = [n for n, _ in dn["accel"]]
    assert stack == [n for n, _ in dp["accel"]] == ["Ax", "Az", "Vy"]
    assert dn["channel_types"] == dp["channel_types"]
    assert dn["channel_types"] == {"Ax": "piezo", "Az": "accel", "Vy": "piezo"}
    for (_, a), (_, b) in zip(dn["accel"], dp["accel"], strict=True):
        assert a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# eps/chi on the full-record L1 file: a pointer to prof, not a traceback
# ---------------------------------------------------------------------------


class TestFullRecordRefused:
    def test_load_channels(self, sn479):
        with pytest.raises(ValueError, match=r"full-record ATOMIX L1 file.*rsi-tpw prof"):
            load_channels(sn479.l1)

    @pytest.mark.parametrize("command", ["eps", "chi"])
    def test_cli(self, sn479, tmp_path, monkeypatch, capsys, command):
        from odas_tpw.rsi.cli import main

        l1 = sn479.l1
        argv = ["rsi-tpw", command, str(l1), "-o", str(tmp_path / command)]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert f"ERROR: {l1.name} is a full-record ATOMIX L1 file" in out
        assert f"rsi-tpw prof {l1.name}" in out


def test_load_from_nc_restores_the_time_base(case):
    """The decimal-day axis snaps to the grid it rounds, anchored at start_time."""
    data = _load_from_nc(case.l1)
    pf = case.pf
    assert _same(data["t_fast"], pf.t_fast)
    assert _same(data["t_slow"], pf.t_slow)
    assert data["t_fast_units"] == f"seconds since {pf.start_time.isoformat()}"
    assert data["start_epoch_s"] == pf.start_time.timestamp()
    assert data["global_attrs"]["instrument_model"] == pf.config["instrument_info"]["model"]


class TestTimeBase:
    def test_snap_only_onto_a_grid(self):
        from odas_tpw.rsi.profile import _snap_to_grid

        grid = np.arange(100) / 64.0
        assert _same(_snap_to_grid(grid + 3e-9, 64.0), grid)
        gapped = grid.copy()
        gapped[50:] += 1.0  # a gap is not rounding
        assert _snap_to_grid(gapped, 64.0) is gapped

    def test_start_time_wins_only_when_it_agrees(self, caplog):
        from datetime import UTC, datetime

        from odas_tpw.rsi.profile import _l1_time_anchor

        origin = ("Days since 2025-01-01T00:00:00Z", 15.5)
        first = datetime(2025, 1, 16, 12, tzinfo=UTC)
        start = "2025-01-16T12:00:00.000400+00:00"  # 0.4 ms off: rounding
        iso, epoch = _l1_time_anchor(origin, start, "f")
        assert iso == start  # verbatim
        assert epoch == pytest.approx(first.timestamp() + 0.0004, abs=1e-6)
        with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.profile"):
            iso, epoch = _l1_time_anchor(origin, "2025-01-16T12:00:05+00:00", "f")
        assert (iso, epoch) == (first.isoformat(), first.timestamp())
        assert "start_time 2025-01-16T12:00:05+00:00 is 5.000 s from" in caplog.text
        assert _l1_time_anchor(("days", 15.5), start, "f") is None
        assert _l1_time_anchor(None, start, "f") is None
