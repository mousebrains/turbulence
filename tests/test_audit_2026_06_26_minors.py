# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Regression tests for MINOR/NIT findings of the 2026-06-26 multi-round review.

Batch 1 — config_base hashing / output-dir defects:
  #84  round(v, 10) collapsed tiny minima (1e-13) to 0.0, corrupting the hash
  #85  the [0-9][0-9] glob missed 3-digit dirs (eps_100), causing collisions
  #89  mixed int/str dict keys raised TypeError during canonicalization

Batch 2 — packaging/config drift:
  #69  mypy per-module overrides listed three nonexistent rsi modules
       (rsi.spectral/ocean/nasmyth), silently suppressing nothing
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

from odas_tpw.config_base import ConfigManager, _normalize_nested, _normalize_value

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _mypy_override_modules() -> list[str]:
    data = tomllib.loads(_PYPROJECT.read_text())
    mods: list[str] = []
    for ov in data["tool"]["mypy"]["overrides"]:
        mods.extend(ov["module"])
    return mods


class TestMypyOverrideModulesExist:
    def test_no_stale_rsi_modules(self):
        # The consolidated rsi.spectral/ocean/nasmyth names were never real
        # modules; the override must not reference them.
        stale = {"odas_tpw.rsi.spectral", "odas_tpw.rsi.ocean", "odas_tpw.rsi.nasmyth"}
        assert stale.isdisjoint(_mypy_override_modules())

    def test_every_override_module_is_importable(self):
        # Each module named in a mypy override must actually exist, or the
        # override silently suppresses nothing (config drift).
        for mod in _mypy_override_modules():
            assert importlib.util.find_spec(mod) is not None, f"missing module: {mod}"


class TestConfigBaseHashing:
    def test_tiny_float_not_collapsed_to_zero(self):
        # round(v, 10) zeroed anything < ~5e-11; significant-figure normalization
        # keeps 1e-13 and 5e-13 distinct.
        assert _normalize_value(1e-13) != 0.0
        mgr = ConfigManager({"epsilon": {"epsilon_minimum": 1e-13}})
        h1 = mgr.compute_hash("epsilon", {"epsilon_minimum": 1e-13})
        h2 = mgr.compute_hash("epsilon", {"epsilon_minimum": 5e-13})
        assert h1 != h2
        # Integers still collapse cleanly (256.0 -> 256), magnitude preserved.
        assert _normalize_value(256.0) == 256

    def test_mixed_type_dict_keys_canonicalize(self):
        # An instrument map with an unquoted numeric serial (int) and a string
        # serial must not raise TypeError: str < int.
        out = _normalize_nested({465: "a", "SN479": "b"})
        assert out == {"465": "a", "SN479": "b"}

    def test_output_dir_sees_three_digit_dirs(self, tmp_path):
        # A 3-digit eps_100 must be found by the scan (and reused for the same
        # config), not missed by a fixed two-digit glob.
        mgr = ConfigManager({"epsilon": {"fft_length": 1024}})
        h = mgr.compute_hash("epsilon", {"fft_length": 1024})
        d100 = tmp_path / "eps_100"
        d100.mkdir()
        (d100 / f".params_sha256_{h}").touch()
        got = mgr.resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 1024})
        assert got.name == "eps_100"

    def test_different_config_does_not_collide_on_existing_dir(self, tmp_path):
        # A different config must NOT be handed an existing dir owned by another
        # config (the atomic-claim / width-adaptive-scan fix).
        mgr = ConfigManager({"epsilon": {"fft_length": 1024}})
        h_a = mgr.compute_hash("epsilon", {"fft_length": 1024})
        d0 = tmp_path / "eps_00"
        d0.mkdir()
        (d0 / f".params_sha256_{h_a}").touch()
        got = mgr.resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 2048})
        assert got.name != "eps_00"
