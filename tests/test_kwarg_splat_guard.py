"""Every `epsilon:`/`chi:` config key must be either excluded or accepted.

Both stages splat their config section into the compute function as keyword
arguments.  A key that is neither stripped nor a real parameter raises
`TypeError` *inside* the per-profile try/except, so the run reports "N file
errors" and writes an EMPTY product directory -- there is no traceback and the
pipeline still exits reporting stage success.

That is exactly how `chi.mixing_use_qc_fallback` (added with the mixing QC
fallback) silently disabled chi for a whole cruise: 1330 profiles, zero chi
files.  These tests pin the invariant for both stages so a new config key
cannot repeat it.
"""

from __future__ import annotations

import inspect

import pytest

from odas_tpw.perturb.config import DEFAULTS, resolve_window_config
from odas_tpw.perturb.pipeline import (
    _CHI_KWARG_EXCLUDE,
    _EPSILON_KWARG_EXCLUDE,
)
from odas_tpw.rsi.chi_io import _compute_chi
from odas_tpw.rsi.dissipation import _compute_epsilon


def _accepted(func) -> set[str]:
    sig = inspect.signature(func)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        pytest.skip(f"{func.__name__} takes **kwargs; the splat cannot leak")
    return set(sig.parameters)


def _splatted(section: str, exclude: tuple[str, ...]) -> set[str]:
    """The keys that actually reach the compute function.

    Mirrors the pipeline: resolve_window_config() first (it strips the
    duration keys and adds the sample counts), then the exclusion list.
    """
    resolved = resolve_window_config(DEFAULTS[section], fs=512.0, section=section)
    return {k for k in resolved if k not in exclude}


@pytest.mark.parametrize(
    ("section", "exclude", "func"),
    [
        ("epsilon", _EPSILON_KWARG_EXCLUDE, _compute_epsilon),
        ("chi", _CHI_KWARG_EXCLUDE, _compute_chi),
    ],
)
def test_no_config_key_leaks_into_the_splat(section, exclude, func):
    leaked = sorted(_splatted(section, exclude) - _accepted(func))
    assert not leaked, (
        f"{section}: config key(s) {leaked} are splatted into "
        f"{func.__name__}() but are not parameters of it. Either add them to "
        f"the exclusion list in perturb/pipeline.py (and consume them "
        f"explicitly) or accept them in the signature. Left as-is, EVERY "
        f"profile raises TypeError and the {section} product is silently empty."
    )


@pytest.mark.parametrize(
    ("section", "exclude"),
    [("epsilon", _EPSILON_KWARG_EXCLUDE), ("chi", _CHI_KWARG_EXCLUDE)],
)
def test_exclusion_list_has_no_dead_entries(section, exclude):
    """A stale exclusion hides a key the callee would now accept."""
    known = set(DEFAULTS[section]) | {"fft_length", "diss_length", "overlap"}
    dead = sorted(set(exclude) - known)
    assert not dead, f"{section}: exclusion list names unknown key(s) {dead}"


def test_mixing_use_qc_fallback_is_stripped_from_the_chi_splat():
    """The specific regression: it is a chi-config key read by the mixing
    step, never a _compute_chi parameter."""
    assert "mixing_use_qc_fallback" in DEFAULTS["chi"]
    assert "mixing_use_qc_fallback" in _CHI_KWARG_EXCLUDE
    assert "mixing_use_qc_fallback" not in inspect.signature(_compute_chi).parameters
