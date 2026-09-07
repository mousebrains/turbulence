"""`instruments.<SN>.*` is hashed only into the stages that consume it.

The block versions a stage's output directory, but not every key touches every
stage. ``fp07_tau_scale`` is read only by the chi computation, so letting it
version the DISS directory forced a full epsilon recompute that could not change
a value -- verified bit-identical over 34820 epsilon / FM / fom / var_resolved
values on ARCTERX-2022. ``exclude_shear_probes`` does reach diss, and reaches chi
transitively through Method 1's ``epsilonMean``, so it must version both.
"""

import hashlib
import json

import pytest

from odas_tpw.perturb.config import (
    _INSTRUMENT_KEYS_BY_STAGE,
    _INSTRUMENT_VALID_KEYS,
    canonical_instruments_for_hash,
    upstream_for,
)

NONE: dict = {}
TAU = {"instruments": {"SN194": {"fp07_tau_scale": {"T1": 0.3}}}}
TAU2 = {"instruments": {"SN194": {"fp07_tau_scale": {"T1": 0.4}}}}
EXC = {"instruments": {"SN194": {"exclude_shear_probes": ["sh2"]}}}
BOTH = {
    "instruments": {
        "SN194": {"exclude_shear_probes": ["sh2"], "fp07_tau_scale": {"T1": 0.3}}
    }
}


def sig(cfg, stage):
    return hashlib.sha256(
        json.dumps(upstream_for(stage, cfg), sort_keys=True, default=str).encode()
    ).hexdigest()


class TestDissIgnoresChiOnlyKeys:
    def test_tau_alone_does_not_reversion_diss(self):
        """The whole point: a chi-side knob must leave diss reusable."""
        assert sig(NONE, "diss") == sig(TAU, "diss")

    def test_tau_added_beside_an_exclude_does_not_reversion_diss(self):
        assert sig(EXC, "diss") == sig(BOTH, "diss")

    def test_changing_tau_does_not_reversion_diss(self):
        assert sig(TAU, "diss") == sig(TAU2, "diss")


class TestChiStillSeesEverything:
    def test_tau_reversions_chi(self):
        assert sig(NONE, "chi") != sig(TAU, "chi")

    def test_a_different_tau_reversions_chi(self):
        """A different tau is a different chi; it must not reuse the old dir."""
        assert sig(TAU, "chi") != sig(TAU2, "chi")

    def test_exclude_reversions_chi_too(self):
        """It reaches chi through Method 1's epsilonMean."""
        assert sig(NONE, "chi") != sig(EXC, "chi")


class TestExcludeStillVersionsDiss:
    def test_exclude_reversions_diss(self):
        assert sig(NONE, "diss") != sig(EXC, "diss")


class TestProjection:
    def test_an_instrument_with_no_relevant_key_is_dropped_entirely(self):
        """Not left as an empty mapping -- `{SN: {}}` would hash differently
        from `{}` and defeat the whole point."""
        out = canonical_instruments_for_hash(
            TAU["instruments"], _INSTRUMENT_KEYS_BY_STAGE["diss"]
        )
        assert out == {}

    def test_relevant_keys_survive_the_projection(self):
        out = canonical_instruments_for_hash(
            BOTH["instruments"], _INSTRUMENT_KEYS_BY_STAGE["chi"]
        )
        assert set(out["SN194"]) == {"exclude_shear_probes", "fp07_tau_scale"}

    def test_no_keys_argument_keeps_everything(self):
        """Back-compat: the unfiltered call is unchanged."""
        out = canonical_instruments_for_hash(BOTH["instruments"])
        assert set(out["SN194"]) == {"exclude_shear_probes", "fp07_tau_scale"}

    def test_every_valid_key_is_assigned_to_at_least_one_stage(self):
        """A new instruments key that reaches no stage would silently fail to
        re-version anything that consumes it."""
        assigned = set().union(*_INSTRUMENT_KEYS_BY_STAGE.values())
        missing = _INSTRUMENT_VALID_KEYS - assigned
        assert not missing, f"instruments keys not mapped to any stage: {sorted(missing)}"

    @pytest.mark.parametrize("stage", sorted(_INSTRUMENT_KEYS_BY_STAGE))
    def test_stage_key_sets_are_subsets_of_the_valid_keys(self, stage):
        assert _INSTRUMENT_KEYS_BY_STAGE[stage] <= _INSTRUMENT_VALID_KEYS
