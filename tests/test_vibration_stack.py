# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""One Goodman vibration stack for every entry point (issue #168).

The adapter (``pfile_to_l1data``, the rsi ``run_pipeline`` route) chose the
vibration channels by sensor TYPE; ``load_channels`` on a ``.p`` file
(``rsi-tpw eps``) or a per-profile NetCDF (perturb), and the quick-look
viewers, chose them by NAME (``^A[xyz]$``); ``nc_to_l1data`` took a
``p_to_L1`` file's ACC alone whenever it had one. They disagreed on the
piezo-typed channels named otherwise: the adapter cleaned shear against them,
the name rule dropped them (silently beside an Ax; with no Ax at all epsilon
refused the file). No instrument we fly is mixed (all piezo-only, named
Ax/Ay), so the fixtures here are SYNTHETIC: the committed SN479 file with its
configuration rewritten.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the viewer test builds a figure; before any pyplot import

import numpy as np
import pytest

from odas_tpw.rsi import config_patch as cp
from odas_tpw.rsi.adapter import pfile_to_l1data
from odas_tpw.rsi.helpers import (
    AC_PATTERN,
    _build_l1data_from_channels,
    load_channels,
    pfile_vibration_channels,
    select_vibration_channels,
    vibration_label,
)
from odas_tpw.rsi.p_file import PFile

SN479 = Path(__file__).parent / "data" / "SN479_0006.p"
needs_sn479 = pytest.mark.skipif(not SN479.exists(), reason="SN479_0006.p test data not available")

# The mixed fixture's stack: SN479's own piezo Ax, a true accelerometer Az,
# and a piezo named outside ^A[xyz]$ (Vy) -- the set the two rules split on.
MIXED_STACK = ["Ax", "Az", "Vy"]
MIXED_TYPES = {"Ax": "piezo", "Az": "accel", "Vy": "piezo"}


def _old_name_rule(fast) -> list[str]:
    """load_channels' and the viewers' selection before #168 (bdd0a4d)."""
    return sorted(n for n in fast if AC_PATTERN.match(n))


def _old_adapter_rule(types, fast) -> tuple[list[str], str]:
    """pfile_to_l1data's selection and label before #168 (bdd0a4d)."""
    acc = sorted(n for n in fast if AC_PATTERN.match(n) and types[n] != "piezo")
    piezo = sorted(n for n, t in types.items() if t == "piezo")
    names = sorted(set(acc) | set(piezo))
    return names, ("ACC" if acc else "VIB") if names else "NONE"


def _old_rules(pf) -> tuple[list[str], tuple[list[str], str]]:
    types = {n: pf.channel_info[n]["type"] for n in pf.channels}
    return _old_name_rule(pf._fast_channels), _old_adapter_rule(types, pf._fast_channels)


def _sn479_with(tmp_path_factory, stem: str, renames, channels) -> Path:
    """A copy of SN479 with channels renamed, then edited by config_patch."""
    if not SN479.exists():
        pytest.skip("SN479_0006.p test data not available")
    text = cp.read_config_text(SN479)
    # config_patch refuses to rename (a channel's name is its key), so rename
    # here, then retype with config_patch.
    for old, new in renames:
        text, n = re.subn(rf"(?m)^(name\s*=\s*){old}(\s*)$", rf"\g<1>{new}\g<2>", text)
        assert n == 1, (old, n)
    spec = cp.EditSpec(note=f"#168 synthetic {stem}", author="test", channels=channels)
    text, changes = cp.edit_config_text(text, spec)
    assert changes
    dst = tmp_path_factory.mktemp(stem) / f"SN479_{stem}_0006.p"
    cp.write_patched_pfile(SN479, dst, text)
    return dst


@pytest.fixture(scope="module")
def mixed_p(tmp_path_factory) -> Path:
    """SN479 with a MIXED vibration config: its own piezo Ax, its piezo Ay
    renamed Vy, and the fast Turbidity column turned into an accelerometer Az."""
    # coef0/coef1 must not be exactly "0"/"1": parse_config rewrites such an
    # accel to piezo (setupstr.m parity), which would leave nothing mixed.
    return _sn479_with(
        tmp_path_factory,
        "mixed",
        renames=(("Ay", "Vy"), ("Turbidity", "Az")),
        channels={"Az": {"type": "accel", "coef0": "2.048", "coef1": "0.4"}},
    )


@pytest.fixture(scope="module")
def slow_piezo_p(tmp_path_factory) -> Path:
    """SN479 with its ground reference (id 0, a SLOW column) typed piezo."""
    return _sn479_with(
        tmp_path_factory, "slowpiezo", renames=(), channels={"Gnd": {"type": "piezo"}}
    )


def _helpers_l1(data, s_slow: int, e_slow: int, ratio: int):
    """L1Data the load_channels route builds for one slow-index slice."""
    n = len(data["t_fast"])
    s_fast, e_fast = s_slow * ratio, min((e_slow + 1) * ratio, n)
    return _build_l1data_from_channels(
        data,
        s_fast,
        e_fast,
        np.full(n, 0.6),
        np.full(n, 10.0),
        np.full(n, 20.0),
        direction="down",
    )


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


# Non-vibration channels; all fast but the slow T1.
OTHER_TYPES = {"sh1": "shear", "sh2": "shear", "T1_dT1": "therm", "T1": "therm"}


class TestSelectVibrationChannels:
    """Types cover every channel, as :func:`pfile_channel_types` does."""

    @staticmethod
    def _select(vib_types, *args, **kwargs):
        types = {**OTHER_TYPES, **vib_types}
        fast = set(types) - {"T1"}
        return select_vibration_channels(types, fast, *args, **kwargs)

    def test_piezo_named_outside_the_pattern_is_kept(self):
        assert self._select({"Ax": "piezo", "Vy": "piezo"}) == (["Ax", "Vy"], "VIB")

    def test_union_with_a_true_accelerometer(self):
        assert self._select(MIXED_TYPES) == (MIXED_STACK, "ACC")

    def test_accelerometers_are_chosen_by_name(self):
        # The adapter's rule: a non-piezo channel counts only when the pattern
        # names it -- an accel-typed "Acc1" is not in the default stack ...
        types = {"Ax": "accel", "Acc1": "accel"}
        assert self._select(types) == (["Ax"], "ACC")
        # ... and a caller's pattern is what admits it.
        assert self._select(types, r"^A(cc\d|[xyz])$") == (["Acc1", "Ax"], "ACC")

    def test_no_pattern_can_drop_a_piezo(self):
        types = {"Ax": "piezo", "Ay": "piezo", "Az": "accel"}
        for pattern in (r"^$", r"^Az$", re.compile(r"^nothing$")):
            names, _ = self._select(types, pattern)
            assert {"Ax", "Ay"} <= set(names), pattern

    def test_untyped_source_falls_back_to_the_name_rule(self):
        # A NetCDF written without sensor_type attrs: the historical selection.
        fast = {"Ax", "Ay", "Az", "Vy", "sh1"}
        assert select_vibration_channels({}, fast) == (["Ax", "Ay", "Az"], "ACC")

    def test_slow_piezo_is_left_out_loudly(self):
        # Pz is in types but not among the fast channels.
        types = {**OTHER_TYPES, "Ax": "piezo", "Pz": "piezo"}
        with pytest.warns(UserWarning, match=r"f\.p: piezo channel\(s\) \['Pz'\] are not fast"):
            names, label = select_vibration_channels(types, {"Ax", "sh1"}, context="f.p")
        assert (names, label) == (["Ax"], "VIB")

    def test_empty(self):
        assert select_vibration_channels({"sh1": "shear"}, {"sh1"}) == ([], "NONE")

    def test_label(self):
        assert vibration_label([], {}) == "NONE"
        assert vibration_label(["Ax", "Vy"], {"Ax": "piezo", "Vy": "piezo"}) == "VIB"
        assert vibration_label(["Ax", "Az"], {"Ax": "piezo", "Az": "accel"}) == "ACC"
        assert vibration_label(["Ax"], {}) == "ACC"  # unknown type: historical label


# ---------------------------------------------------------------------------
# Piezo-only Ax/Ay -- every instrument we fly -- selects what it always did
# ---------------------------------------------------------------------------


class TestPiezoOnlyUnchanged:
    @pytest.mark.parametrize(
        "types",
        [
            {"Ax": "piezo", "Ay": "piezo"},  # MR / modern VMP (SN479, MR330, osu685)
            {"Ax": "piezo", "Ay": "piezo", "Az": "piezo"},
            {"Ax": "accel", "Ay": "accel"},  # an old VMP with real accelerometers
            {},  # no vibration sensor at all
        ],
    )
    def test_synthetic_matches_both_old_rules(self, types):
        fast = set(types) | {"sh1", "sh2"}
        all_types = {**types, "sh1": "shear", "sh2": "shear", "T1": "therm"}
        names, label = select_vibration_channels(all_types, fast)
        assert names == _old_name_rule(fast)
        assert (names, label) == _old_adapter_rule(all_types, fast)

    @needs_sn479
    def test_sn479_selects_what_both_old_rules_did(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = PFile(SN479)
        assert pfile_vibration_channels(pf) == (["Ax", "Ay"], "VIB")
        assert _old_rules(pf) == (["Ax", "Ay"], (["Ax", "Ay"], "VIB"))

        data = load_channels(pf)
        assert [n for n, _ in data["accel"]] == ["Ax", "Ay"]
        for name, arr in data["accel"]:
            np.testing.assert_array_equal(arr, pf.channels[name])

        ratio = round(pf.fs_fast / pf.fs_slow)
        l1 = pfile_to_l1data(pf, profile_slice=(2000, 6000), speed=0.6)
        expect = np.stack([pf.channels[n][2000 * ratio : 6001 * ratio] for n in ("Ax", "Ay")])
        assert l1.vib_type == "VIB"
        np.testing.assert_array_equal(l1.vib, expect)


# ---------------------------------------------------------------------------
# A mixed piezo + accelerometer config: identical through every entry point
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pf(mixed_p) -> PFile:
    """The mixed file, read once for the tests that need a PFile."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PFile(mixed_p)


class TestMixedConfigEveryEntryPoint:
    def test_fixture_is_mixed_and_exercises_the_old_split(self, pf):
        assert {n: pf.channel_info[n]["type"] for n in MIXED_STACK} == MIXED_TYPES
        assert all(pf.is_fast(n) for n in MIXED_STACK)
        # Before #168 the two rules disagreed on this config: Vy is the set.
        assert _old_rules(pf) == (["Ax", "Az"], (MIXED_STACK, "ACC"))
        assert pfile_vibration_channels(pf) == (MIXED_STACK, "ACC")

    def test_adapter_and_load_channels_build_identical_stacks(self, pf, mixed_p):
        s_slow, e_slow = 2000, 6000
        ratio = round(pf.fs_fast / pf.fs_slow)
        l1_adapter = pfile_to_l1data(pf, profile_slice=(s_slow, e_slow), speed=0.6)

        # The .p route reads the file itself (its own PFile), as rsi-tpw eps does.
        data = load_channels(mixed_p)
        assert [n for n, _ in data["accel"]] == MIXED_STACK
        assert data["channel_types"] == MIXED_TYPES
        l1_helpers = _helpers_l1(data, s_slow, e_slow, ratio)

        assert l1_adapter.vib_type == l1_helpers.vib_type == "ACC"
        assert l1_adapter.vib.shape == (3, (e_slow - s_slow + 1) * ratio)
        assert l1_adapter.vib.tobytes() == l1_helpers.vib.tobytes()
        s_fast, e_fast = s_slow * ratio, (e_slow + 1) * ratio
        expect = np.stack([pf.channels[n][s_fast:e_fast] for n in MIXED_STACK])
        np.testing.assert_array_equal(l1_adapter.vib, expect)

    def test_per_profile_netcdf_route_builds_the_same_stack(self, pf, mixed_p, tmp_path):
        import netCDF4

        from odas_tpw.rsi.profile import extract_profiles

        prof = extract_profiles(mixed_p, tmp_path)[0]
        with netCDF4.Dataset(str(prof)) as ds:
            s_slow = int(ds.profile_start_index_slow)
            e_slow = int(ds.profile_end_index_slow)
        data = load_channels(prof)
        assert [n for n, _ in data["accel"]] == MIXED_STACK
        assert data["channel_types"] == MIXED_TYPES

        ratio = round(pf.fs_fast / pf.fs_slow)
        l1_nc = _helpers_l1(data, 0, e_slow - s_slow, ratio)
        l1_adapter = pfile_to_l1data(pf, profile_slice=(s_slow, e_slow), speed=0.6)
        assert l1_nc.vib_type == l1_adapter.vib_type == "ACC"
        # The per-profile file stores float32: the same stack, at its precision.
        np.testing.assert_array_equal(l1_nc.vib, l1_adapter.vib.astype(np.float32))

    def test_full_record_l1_netcdf_route_builds_the_same_stack(self, pf, mixed_p, tmp_path):
        import netCDF4

        from odas_tpw.rsi.adapter import nc_to_l1data
        from odas_tpw.rsi.convert import p_to_L1

        _, nc_path = p_to_L1(mixed_p, tmp_path / "L1.nc")
        # The file records sensors by ROLE; the reader applies the rule.
        with netCDF4.Dataset(str(nc_path)) as ds:
            g = ds.groups["L1_converted"]
            assert g.variables["VIB"].sensor_names == "Ax, Vy"
            assert g.variables["ACC"].sensor_names == "Az"
        l1_nc = nc_to_l1data(nc_path)
        l1_adapter = pfile_to_l1data(pf, speed=0.6)
        assert l1_nc.vib_type == l1_adapter.vib_type == "ACC"
        assert l1_nc.vib.shape == l1_adapter.vib.shape == (3, len(pf.t_fast))
        assert l1_nc.vib.tobytes() == l1_adapter.vib.tobytes()

    def test_quick_look_viewer_uses_the_same_stack(self, pf):
        import matplotlib.pyplot as plt

        from odas_tpw.rsi.diss_look import DissLookViewer

        try:
            viewer = DissLookViewer(pf, fft_length=256, f_AA=98.0)
            assert [n for n, _ in viewer.accel] == MIXED_STACK
        finally:
            plt.close("all")

    def test_accel_pattern_narrows_accelerometers_never_piezos(self, mixed_p):
        # A pattern that names no accelerometer drops Az, but both piezos stay.
        data = load_channels(mixed_p, accel_pattern=r"^Ax$")
        assert [n for n, _ in data["accel"]] == ["Ax", "Vy"]
        assert vibration_label(["Ax", "Vy"], data["channel_types"]) == "VIB"


# ---------------------------------------------------------------------------
# The full-record L1 NetCDF reader on files it did not write
# ---------------------------------------------------------------------------


def _write_l1(path: Path, blocks, n_time: int = 64) -> Path:
    """An L1_converted group holding VIB/ACC blocks: (var, names or None, n)."""
    import netCDF4

    with netCDF4.Dataset(str(path), "w") as ds:
        g = ds.createGroup("L1_converted")
        g.createDimension("TIME", n_time)
        g.createVariable("TIME", "f8", ("TIME",))[:] = np.arange(n_time) / 512.0
        g.createVariable("PRES", "f8", ("TIME",))[:] = np.linspace(5.0, 50.0, n_time)
        for k, (var, names, n) in enumerate(blocks):
            g.createDimension(f"N_{var}", n)
            v = g.createVariable(var, "f8", (f"N_{var}", "TIME"))
            v[:] = 100.0 * k + np.arange(n)[:, None] + np.zeros(n_time)
            if names is not None:
                v.sensor_names = ", ".join(names)
    return path


class TestFullRecordL1Reader:
    def test_named_rows_go_through_the_rule(self, tmp_path):
        from odas_tpw.rsi.adapter import nc_to_l1data

        # An accelerometer named outside ^A[xyz]$ (Acc1) is not in the stack,
        # as in every other route; the rows come out name-sorted.
        nc_path = _write_l1(
            tmp_path / "named.nc", [("VIB", ["Vy", "Ax"], 2), ("ACC", ["Az", "Acc1"], 2)]
        )
        l1 = nc_to_l1data(nc_path)
        assert l1.vib_type == "ACC"
        # VIB rows 0..1 = Vy, Ax; ACC rows 100..101 = Az, Acc1.
        np.testing.assert_array_equal(l1.vib[:, 0], [1.0, 100.0, 0.0])  # Ax, Az, Vy

    def test_unnamed_rows_are_all_kept_vib_first(self, tmp_path):
        from odas_tpw.rsi.adapter import nc_to_l1data

        # Without sensor_names the name rule cannot run: no row is dropped
        # (this reader used to return ACC alone).
        nc_path = _write_l1(tmp_path / "unnamed.nc", [("VIB", None, 2), ("ACC", None, 1)])
        l1 = nc_to_l1data(nc_path)
        assert l1.vib_type == "ACC"
        np.testing.assert_array_equal(l1.vib[:, 0], [0.0, 1.0, 100.0])

    def test_unnamed_piezo_only_is_vib(self, tmp_path):
        from odas_tpw.rsi.adapter import nc_to_l1data

        l1 = nc_to_l1data(_write_l1(tmp_path / "vib.nc", [("VIB", None, 2)]))
        assert (l1.vib_type, l1.n_vib) == ("VIB", 2)


# ---------------------------------------------------------------------------
# A piezo on a slow column: left out loudly, never a crash
# ---------------------------------------------------------------------------


class TestSlowPiezo:
    MATCH = r"piezo channel\(s\) \['Gnd'\] are not fast-rate"

    @staticmethod
    def _assert_points_here(record):
        hits = [w for w in record if "['Gnd']" in str(w.message)]
        assert hits, [str(w.message) for w in record]
        # stacklevel: attributed to the code that called the entry point.
        assert all(w.filename == __file__ for w in hits), [w.filename for w in hits]

    def test_adapter_leaves_it_out(self, slow_piezo_p, caplog):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = PFile(slow_piezo_p)
        assert pf.channel_info["Gnd"]["type"] == "piezo" and not pf.is_fast("Gnd")
        with (
            caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.helpers"),
            pytest.warns(UserWarning, match=self.MATCH) as record,
        ):
            l1 = pfile_to_l1data(pf, profile_slice=(2000, 6000), speed=0.6)
        # (main crashed here: np.stack of a slow slice with the fast ones)
        assert (l1.n_vib, l1.vib_type) == (2, "VIB")  # Ax, Ay
        self._assert_points_here(record)
        assert "SN479_slowpiezo_0006.p: piezo channel(s) ['Gnd']" in caplog.text

    def test_load_channels_both_routes_leave_it_out(self, slow_piezo_p, tmp_path):
        from odas_tpw.rsi.profile import extract_profiles

        with pytest.warns(UserWarning, match=self.MATCH) as record:
            data = load_channels(slow_piezo_p)
        assert [n for n, _ in data["accel"]] == ["Ax", "Ay"]
        self._assert_points_here(record)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prof = extract_profiles(slow_piezo_p, tmp_path)[0]
        with pytest.warns(UserWarning, match=self.MATCH) as record:
            data = load_channels(prof)
        assert [n for n, _ in data["accel"]] == ["Ax", "Ay"]
        self._assert_points_here(record)
