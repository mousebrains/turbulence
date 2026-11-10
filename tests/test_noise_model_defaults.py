# Nov-2026, Claude and Pat Welch, pat@mousebrains.com
"""chi's FP07 noise model on generic defaults is never silent (#282).

The noise model takes ``e_b``, ``b``, ``g``, ``adc_fs``, ``adc_bits``, ``beta_1``
and ``t_0`` from the thermistor's configuration section, and replaced any it did
not find by ``FP07NoiseConfig``'s defaults without a word. A section typed other
than ``therm`` carries none: the Taiwan 2013 VMP's FP07s are ``poly``, an in-situ
quadratic. chi now warns, stamps ``fp07_noise_defaults_<channel>`` on the
product, and perturb adds a run-summary notice. The numbers do not change.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import netCDF4
import pytest

from odas_tpw.rsi import config_patch as cp
from odas_tpw.rsi.chi_io import (
    NOISE_DEFAULTS_ATTR,
    NOISE_MODEL_KEYS,
    _extract_therm_cal,
    _noise_defaults,
    compute_chi_file,
)
from odas_tpw.rsi.profile import extract_profiles

DATA = Path(__file__).parent / "data"
SN479 = DATA / "SN479_0006.p"
needs_sn479 = pytest.mark.skipif(not SN479.exists(), reason="SN479 test data not available")

ALL = " ".join(NOISE_MODEL_KEYS)
SPELLED = "e_b, b, g, adc_fs, adc_bits, beta_1, t_0"  # as the configuration spells them
_FP07 = ("T1", "T2", "T1_dT1", "T2_dT2")

# SN479's embedded T1 stanza (the electronics plus a nominal bead).
_CFG = {
    "name": "T1", "type": "therm", "e_b": "0.68283", "b": "0.99861", "g": "6.0",
    "beta_1": "3143.55", "beta_2": "2.5e5", "adc_fs": "4.096", "adc_bits": "16",
    "t_0": "289.301",
}


class TestWhichKeys:
    def test_a_full_therm_section_supplies_them_all(self):
        assert _noise_defaults(_extract_therm_cal(_CFG)) == []

    def test_a_poly_section_supplies_none(self):
        poly = {"name": "T1", "type": "poly", "coef0": "12.9", "coef1": "1.1e-3"}
        assert _noise_defaults(_extract_therm_cal(poly)) == list(NOISE_MODEL_KEYS)

    def test_one_missing_key_is_named(self):
        cfg = {k: v for k, v in _CFG.items() if k != "e_b"}
        assert _noise_defaults(_extract_therm_cal(cfg)) == ["e_b"]

    def test_beta_2_is_not_required(self):
        cfg = {k: v for k, v in _CFG.items() if k != "beta_2"}
        assert _noise_defaults(_extract_therm_cal(cfg)) == []

    def test_the_legacy_beta_key_counts_as_beta_1(self):
        cfg = {k: v for k, v in _CFG.items() if k != "beta_1"} | {"beta": "3143.55"}
        assert _noise_defaults(_extract_therm_cal(cfg)) == []

    def test_an_in_situ_fit_supplies_the_bead(self):
        electronics = {k: v for k, v in _CFG.items() if k not in ("t_0", "beta_1", "beta_2")}
        fit = {"t_0": 288.0, "beta_1": 3050.0}
        assert _noise_defaults(_extract_therm_cal(electronics)) == ["beta_1", "T_0"]
        assert _noise_defaults(_extract_therm_cal(electronics, fit)) == []


# The therm keys a "poly" section does not carry (Taiwan 2013's hold only id, name,
# type, coef0-2, diff_gain and units).
_THERM_KEYS = {"a", "b", "e_b", "g", "adc_fs", "adc_bits", "adc_zero", "beta", "beta_1",
               "beta_2", "beta_3", "t_0"}


def _strip_therm_keys(text: str, names: tuple[str, ...]) -> str:
    """*text* without the therm keys of the channel sections named *names*."""
    out: list[str] = []
    section: list[str] = []

    def flush() -> None:
        name = next((ln.split("=", 1)[1].strip() for ln in section
                     if ln.split("=", 1)[0].strip().lower() == "name" and "=" in ln), None)
        for ln in section:
            key = ln.split("=", 1)[0].strip().lower() if "=" in ln else None
            if name in names and key in _THERM_KEYS and not ln.lstrip().startswith(";"):
                continue
            out.append(ln)
        section.clear()

    for ln in text.splitlines(keepends=True):
        if ln.lstrip().startswith("["):
            flush()
        section.append(ln)
    flush()
    return "".join(out)


@pytest.fixture(scope="module")
def poly_p(tmp_path_factory) -> Path:
    """SN479 with its FP07 channels typed "poly" (a linear counts -> C) and no
    therm keys left, as the Taiwan 2013 VMP's are after their in-situ fit."""
    d = tmp_path_factory.mktemp("poly")
    linear = {"type": "poly", "coef0": "20.0", "coef1": "1e-4"}
    spec = cp.EditSpec(
        note="FP07 typed poly", author="test", channels={n: dict(linear) for n in _FP07}
    )
    text, changes = cp.edit_config_text(
        cp.read_config_text(SN479), spec, add_keys=True, when="2026-11-09 00:00:00"
    )
    assert changes
    text = _strip_therm_keys(text, _FP07)
    vmp = d / "VMP"
    vmp.mkdir()
    p = cp.write_patched_pfile(SN479, vmp / SN479.name, text)
    from odas_tpw.rsi.p_file import PFile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = {c["name"]: c for c in PFile(p).config["channels"] if c.get("name") in _FP07}
    for name, section in cfg.items():  # the fixture is what it says
        assert section["type"] == "poly", name
        assert not (_THERM_KEYS & set(section)), (name, sorted(_THERM_KEYS & set(section)))
    return p


def _chi(src: Path, out: Path, caplog) -> tuple[dict, list[str]]:
    """The first chi product's global attrs and the #282 warnings it logged."""
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.chi_io"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = compute_chi_file(src, out)[0]
    with netCDF4.Dataset(str(path)) as ds:
        attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}
    return attrs, [r.getMessage() for r in caplog.records if "#282" in r.getMessage()]


def _marks(attrs: dict) -> dict:
    return {k: v for k, v in attrs.items() if k.startswith(f"{NOISE_DEFAULTS_ATTR}_")}


@needs_sn479
class TestChi:
    def test_the_p_route_warns_and_marks(self, poly_p, tmp_path, caplog):
        attrs, warned = _chi(poly_p, tmp_path, caplog)
        assert _marks(attrs) == {
            f"{NOISE_DEFAULTS_ATTR}_T1_dT1": ALL,
            f"{NOISE_DEFAULTS_ATTR}_T2_dT2": ALL,
        }
        assert len(warned) == 1
        # the section the keys are read from, in the configuration's spelling
        assert f"found no {SPELLED} in sections T1, T2 (read for T1_dT1, T2_dT2)" in warned[0]

    def test_the_profile_route_marks_the_same(self, poly_p, tmp_path, caplog):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prof = extract_profiles(poly_p, tmp_path / "prof")[0]
        attrs, warned = _chi(prof, tmp_path / "chi", caplog)
        assert _marks(attrs) == {
            f"{NOISE_DEFAULTS_ATTR}_T1_dT1": ALL,
            f"{NOISE_DEFAULTS_ATTR}_T2_dT2": ALL,
        }
        assert len(warned) == 1

    def test_a_calibrated_file_is_neither_marked_nor_warned(self, tmp_path, caplog):
        attrs, warned = _chi(SN479, tmp_path, caplog)
        assert _marks(attrs) == {}
        assert warned == []


class TestRunSummary:
    def test_tally_reads_the_marks(self):
        from odas_tpw.perturb.pipeline import _tally_noise_defaults

        result: dict = {}
        _tally_noise_defaults(result, {"other": 1})
        assert "fp07_noise_defaults" not in result
        _tally_noise_defaults(result, {f"{NOISE_DEFAULTS_ATTR}_T1_dT1": ALL})
        assert result["fp07_noise_defaults"] == {"T1_dT1": ALL}

    def test_files_group_by_what_was_defaulted(self):
        from odas_tpw.perturb.pipeline import _noise_default_notices

        both = {"T1_dT1": ALL, "T2_dT2": ALL}
        lines = _noise_default_notices(
            {"b.p": both, "a.p": both, "c.p": {"T1_dT1": "e_b"}, "d.p": {},
             "e.p": {"T1_dT1": "e_b", "T2_dT2": ALL}}
        )
        assert len(lines) == 3
        assert lines[0].startswith("FP07 noise model: 2 file(s)")
        assert lines[0].endswith("Files: a.p, b.p")
        assert f"no {SPELLED} in sections T1, T2 (read for T1_dT1, T2_dT2) (a" in lines[0]
        assert lines[1].startswith(
            "FP07 noise model: 1 file(s) gave chi's noise model no e_b in section T1 "
            "(read for T1_dT1) (a"
        )
        assert (
            f"no e_b in section T1 (read for T1_dT1); {SPELLED} in section T2 "
            "(read for T2_dT2) (a"
            in lines[2]
        )
        assert _noise_default_notices({}) == []

    def test_a_plain_thermistor_is_its_own_section(self):
        from odas_tpw.rsi.chi_io import describe_noise_defaults

        assert describe_noise_defaults({"T1": ["e_b", "T_0"]}) == "e_b, t_0 in section T1"

    @needs_sn479
    def test_perturb_end_to_end(self, poly_p, tmp_path):
        """The notice on the run that processed the file; the mark on its products,
        which a rerun that skips the file keeps (and names no file)."""
        from odas_tpw.perturb import pipeline as pl

        out = tmp_path / "out"

        def run():
            return pl.run_pipeline({
                "files": {"p_file_root": str(poly_p.parent.parent), "output_root": str(out),
                          "trim": False, "merge": False},
                "chi": {"enable": True}, "ctd": {"enable": False}, "parallel": {"jobs": 1},
            }, p_files=[poly_p])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = run()
        assert first.ok, first.summary()
        lines = [n for n in first.notices if n.startswith("FP07 noise model")]
        assert len(lines) == 1 and lines[0].startswith("FP07 noise model: 1 file(s)")
        assert "t_0 in sections T1, T2 (read for T1_dT1, T2_dT2)" in lines[0]
        assert lines[0].endswith(poly_p.stem)  # the file's output stem

        def marks():
            chi_files = sorted((out / "chi_00").glob("*.nc"))
            assert chi_files
            for path in chi_files:
                with netCDF4.Dataset(str(path)) as ds:
                    assert ds.getncattr(f"{NOISE_DEFAULTS_ATTR}_T1_dT1") == ALL
                    assert ds.getncattr(f"{NOISE_DEFAULTS_ATTR}_T2_dT2") == ALL
            return [p.stat().st_mtime_ns for p in chi_files]

        written = marks()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            again = run()
        assert again.ok
        assert not [n for n in again.notices if n.startswith("FP07 noise model")]
        assert marks() == written  # skipped, not rewritten
