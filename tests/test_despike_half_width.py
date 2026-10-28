"""Shear despike half-width: the `despike_half_width` config key.

It was hard-wired at 0.04 s in ``_compute_epsilon``. These tests pin what the
value removes, that it reaches ``L2Params.despike_sh[2]`` from both config
front ends, that bad values fail at load time, and that the 0.04 default
leaves every signature -- and every epsilon value -- exactly as it was.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from odas_tpw.rsi.dissipation import (
    DESPIKE_HALF_WIDTH,
    DESPIKE_HALF_WIDTH_MAX,
    _compute_epsilon,
    validate_despike_half_width,
)
from odas_tpw.scor160.despike import despike

SAMPLE = Path(__file__).parent / "data" / "SN479_0006.p"


class TestValidate:
    @pytest.mark.parametrize("value", [0.04, 0.01, 0.001, DESPIKE_HALF_WIDTH_MAX, np.float64(0.02)])
    def test_accepts_seconds(self, value):
        assert validate_despike_half_width(value) == float(value)

    @pytest.mark.parametrize(
        "value",
        [0, 0.0, -0.01, 0.51, 1, 20, float("nan"), float("inf"), True, "0.04", None, [0.04]],
    )
    def test_refuses(self, value):
        # >= 1 would silently become SAMPLES in process_l2; a bool is an int.
        with pytest.raises(ValueError, match="despike_half_width"):
            validate_despike_half_width(value)


class TestWhatItRemoves:
    """N//2 samples before the spike, 2*(N//2) after, N = round(hw * fs)."""

    @pytest.mark.parametrize(("half_width", "n_replaced"), [(0.04, 31), (0.02, 16), (0.01, 7)])
    def test_samples_replaced_around_one_spike(self, half_width, n_replaced):
        fs = 512.0
        rng = np.random.default_rng(3)
        x = 0.01 * rng.standard_normal(20 * int(fs))
        x[5000] += 5.0
        y, spikes, _, _ = despike(x, fs, N=round(half_width * fs))
        changed = np.flatnonzero(y != x)
        assert spikes.tolist() == [5000]
        assert len(changed) == n_replaced
        # Asymmetric: the tail gets twice the lead.
        n_half = round(half_width * fs) // 2
        assert changed[0] == 5000 - n_half
        assert changed[-1] == 5000 + 2 * n_half


class TestThreading:
    def _captured_half_width(self, monkeypatch, **kwargs) -> float:
        import odas_tpw.scor160.l2 as l2

        seen: list[np.ndarray] = []

        class _Stop(Exception):
            pass

        def spy(l1, params):
            seen.append(params.despike_sh.copy())
            raise _Stop

        monkeypatch.setattr(l2, "process_l2", spy)
        with pytest.raises(_Stop):
            _compute_epsilon(SAMPLE, **kwargs)
        return float(seen[0][2])

    def test_default_is_the_old_hard_wired_value(self, monkeypatch):
        assert DESPIKE_HALF_WIDTH == 0.04
        assert self._captured_half_width(monkeypatch) == 0.04

    def test_value_reaches_l2params(self, monkeypatch):
        assert self._captured_half_width(monkeypatch, despike_half_width=0.01) == 0.01

    def test_bad_value_fails_before_any_file_access(self, tmp_path):
        with pytest.raises(ValueError, match="despike_half_width"):
            _compute_epsilon(tmp_path / "does_not_exist.p", despike_half_width=2)

    def test_default_is_bit_identical_and_narrower_changes_epsilon(self):
        if not SAMPLE.exists():
            pytest.skip("Test data not available")
        base = _compute_epsilon(SAMPLE)
        same = _compute_epsilon(SAMPLE, despike_half_width=0.04)
        narrow = _compute_epsilon(SAMPLE, despike_half_width=0.01)
        assert len(base) == len(same) == len(narrow) >= 1
        for a, b in zip(base, same):
            np.testing.assert_array_equal(a["epsilon"].values, b["epsilon"].values)
        # The sample file's shear has detected spikes, so 0.01 s must move
        # something; otherwise this test would not show the value is used.
        assert any(
            not np.array_equal(a["epsilon"].values, c["epsilon"].values, equal_nan=True)
            for a, c in zip(base, narrow)
        )


@pytest.mark.parametrize("front_end", ["rsi", "perturb"])
class TestConfig:
    def _cfg(self, front_end):
        if front_end == "rsi":
            from odas_tpw.rsi import config
        else:
            from odas_tpw.perturb import config  # type: ignore[no-redef]
        return config

    def test_in_defaults_and_template(self, front_end, tmp_path):
        cfg = self._cfg(front_end)
        assert cfg.DEFAULTS["epsilon"]["despike_half_width"] == 0.04
        path = cfg.generate_template(tmp_path / "config.yaml")
        assert "despike_half_width: 0.04" in path.read_text()

    def test_default_leaves_the_signature_unchanged(self, front_end):
        cfg = self._cfg(front_end)
        bare = cfg.canonicalize("epsilon", {})
        assert "despike_half_width" not in bare
        assert cfg.compute_hash("epsilon", {"despike_half_width": 0.04}) == cfg.compute_hash(
            "epsilon", {}
        )
        assert cfg.compute_hash("epsilon", {"despike_half_width": 0.01}) != cfg.compute_hash(
            "epsilon", {}
        )

    @pytest.mark.parametrize("bad", ["0", "-0.01", "0.75", "2", "true", "'0.04'"])
    def test_bad_value_fails_at_load(self, front_end, tmp_path, bad):
        cfg = self._cfg(front_end)
        path = tmp_path / "config.yaml"
        path.write_text(f"epsilon:\n  despike_half_width: {bad}\n")
        with pytest.raises(ValueError, match="despike_half_width"):
            cfg.load_config(path)

    def test_good_value_loads(self, front_end, tmp_path):
        cfg = self._cfg(front_end)
        path = tmp_path / "config.yaml"
        path.write_text("epsilon:\n  despike_half_width: 0.01\n")
        assert cfg.load_config(path)["epsilon"]["despike_half_width"] == 0.01
