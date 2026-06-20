# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared configuration management logic.

Provides load/validate/merge/hash/resolve functions parameterized by
a DEFAULTS dict.  Used by both rsi/config.py and perturb/config.py to
eliminate duplication of 9 identical config management functions.
"""

from __future__ import annotations

import glob as globmod
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

from ruamel.yaml import YAML


def _normalize_value(v):
    """Normalize a single value for deterministic JSON encoding."""
    if v is None:
        return None
    # bool check must come before int (bool is a subclass of int)
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        # int(nan) raises ValueError and int(inf) raises OverflowError, which
        # would crash canonicalize -> compute_hash -> resolve_output_dir with an
        # opaque message. Hash non-finite values by their repr instead.
        if not math.isfinite(v):
            return repr(v)
        if v == int(v):
            return int(v)
        return round(v, 10)
    if isinstance(v, str):
        return v
    return v


def _normalize_nested(v):
    """Normalize nested JSON-like values for deterministic hashing."""
    if isinstance(v, list):
        return [_normalize_nested(item) for item in v]
    if isinstance(v, tuple):
        return [_normalize_nested(item) for item in v]
    if isinstance(v, dict):
        return {str(k): _normalize_nested(val) for k, val in sorted(v.items())}
    return _normalize_value(v)


class ConfigManager:
    """Config management parameterized by a DEFAULTS dict.

    Parameters
    ----------
    defaults : dict[str, dict]
        Canonical defaults — one dict per processing section.
    hash_exclude_keys : frozenset[str]
        Keys omitted from canonicalization/hashing (e.g. ``{"diagnostics"}``).
    dynamic_key_sections : frozenset[str]
        Sections whose keys are user-defined at runtime (e.g. instrument
        serial numbers). Strict unknown-key validation is skipped for these
        sections; the caller is responsible for validating the inner
        structure where it is consumed.
    """

    def __init__(
        self,
        defaults: dict[str, dict],
        *,
        hash_exclude_keys: frozenset[str] = frozenset(),
        dynamic_key_sections: frozenset[str] = frozenset(),
    ) -> None:
        self.defaults = defaults
        self.valid_sections = frozenset(defaults)
        self.hash_exclude_keys = hash_exclude_keys
        self.dynamic_key_sections = dynamic_key_sections

    # -- Load / validate ---------------------------------------------------

    def load_config(self, path: str | Path) -> dict[str, dict]:
        """Load a YAML config file and return as plain dict of dicts.

        Raises FileNotFoundError if *path* does not exist, ValueError if
        the file contains unknown sections or keys.
        """
        yaml = YAML()
        with open(path, encoding="utf-8") as fh:
            raw = yaml.load(fh)
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Config file must be a mapping of sections, got "
                f"{type(raw).__name__}"
            )
        # Each section must itself be a mapping (key: value). A list/scalar
        # section otherwise hits dict(v) with an opaque TypeError (#29).
        config: dict[str, dict] = {}
        for k, v in raw.items():
            if v is None:
                config[str(k)] = {}
            elif isinstance(v, Mapping):
                config[str(k)] = dict(v)
            else:
                raise ValueError(
                    f"Config section {str(k)!r} must be a mapping of "
                    f"key: value pairs, got {type(v).__name__}"
                )
        self.validate_config(config)
        return config

    def validate_config(self, config: dict[str, dict]) -> None:
        """Raise ValueError if *config* has unknown sections or keys."""
        for section, params in config.items():
            if section not in self.valid_sections:
                raise ValueError(
                    f"Unknown config section: {section!r}. "
                    f"Valid sections: {sorted(self.valid_sections)}"
                )
            if section in self.dynamic_key_sections:
                continue
            valid_keys = set(self.defaults[section])
            unknown = set(params) - valid_keys
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in [{section}]: {sorted(unknown)}. "
                    f"Valid keys: {sorted(valid_keys)}"
                )

    # -- Three-way merge ---------------------------------------------------

    def merge_config(
        self,
        section: str,
        file_values: dict | None = None,
        cli_overrides: dict | None = None,
    ) -> dict:
        """Merge defaults <- config-file values <- CLI overrides.

        None values in either layer are treated as "not specified" and do
        not mask earlier layers.  Returns a clean kwargs dict with all
        None values removed.
        """
        if section not in self.defaults:
            raise ValueError(f"Unknown section: {section!r}")

        merged = dict(self.defaults[section])

        if file_values:
            for k, v in file_values.items():
                if k in merged and v is not None:
                    merged[k] = v

        if cli_overrides:
            for k, v in cli_overrides.items():
                if k in merged and v is not None:
                    merged[k] = v

        return {k: v for k, v in merged.items() if v is not None}

    # -- Canonicalization and hashing --------------------------------------

    def _canonicalize_section(self, section: str, params: dict) -> dict:
        """Canonicalize a single section's parameters into a normalized dict."""
        if section in self.dynamic_key_sections:
            return {
                str(k): _normalize_nested(v)
                for k, v in sorted((params or {}).items())
                if k not in self.hash_exclude_keys
            }

        base = dict(self.defaults[section])
        for k, v in params.items():
            if k in base and v is not None:
                base[k] = v
        # _normalize_nested (not _normalize_value) so list/dict-valued params
        # (e.g. speed.amplitude_quantile = [1.0, 99.0]) have their nested
        # scalars type-normalized too. _normalize_value passes containers
        # through untouched, so [1.0, 99.0] and [1, 99] would hash differently
        # -> different output dirs / spurious recompute. Matches the
        # dynamic-key branch above. For scalars the two are identical.
        return {
            k: _normalize_nested(v)
            for k, v in sorted(base.items())
            if k not in self.hash_exclude_keys
        }

    def canonicalize(
        self,
        section: str,
        params: dict,
        upstream: list[tuple[str, dict]] | None = None,
    ) -> str:
        """Produce a deterministic JSON string for hashing.

        Overlays *params* on the section defaults, normalizes types, and
        returns compact sorted JSON.
        """
        sections = {}
        if upstream:
            for up_section, up_params in upstream:
                sections[up_section] = self._canonicalize_section(up_section, up_params)
        sections[section] = self._canonicalize_section(section, params)
        return json.dumps(sections, sort_keys=True, separators=(",", ":"))

    def compute_hash(
        self,
        section: str,
        params: dict,
        upstream: list[tuple[str, dict]] | None = None,
    ) -> str:
        """SHA-256 hex digest of the canonical representation."""
        canonical = self.canonicalize(section, params, upstream=upstream)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # -- Output directory management ---------------------------------------

    def resolve_output_dir(
        self,
        base: str | Path,
        prefix: str,
        section: str,
        params: dict,
        upstream: list[tuple[str, dict]] | None = None,
    ) -> Path:
        """Find or create a sequential output directory matching *params*.

        Scans ``base/{prefix}_NN/`` directories for a hash-matching
        signature file.  If found, returns that directory.  Otherwise
        creates the next sequential directory and writes the signature.
        """
        base = Path(base)
        target_hash = self.compute_hash(section, params, upstream=upstream)

        max_seq = -1
        pattern = str(base / f"{prefix}_[0-9][0-9]")
        for d in sorted(globmod.glob(pattern)):
            dp = Path(d)
            try:
                seq = int(dp.name.split("_")[-1])
            except ValueError:
                continue
            max_seq = max(max_seq, seq)
            if not dp.is_dir():
                continue
            sig_file = dp / f".params_sha256_{target_hash}"
            if sig_file.exists():
                return dp

        next_seq = max_seq + 1
        new_dir = base / f"{prefix}_{next_seq:02d}"
        new_dir.mkdir(parents=True, exist_ok=True)
        self.write_signature(new_dir, section, params, upstream=upstream)
        return new_dir

    def write_signature(
        self,
        directory: Path,
        section: str,
        params: dict,
        upstream: list[tuple[str, dict]] | None = None,
    ) -> Path:
        """Write a ``.params_sha256_<hash>`` signature file."""
        h = self.compute_hash(section, params, upstream=upstream)
        canonical = self.canonicalize(section, params, upstream=upstream)
        sig_file = directory / f".params_sha256_{h}"
        sig_file.write_text(canonical)
        return sig_file

    def write_resolved_config(
        self,
        directory: Path,
        section: str,
        params: dict,
        upstream: list[tuple[str, dict]] | None = None,
    ) -> Path:
        """Write a human-readable ``config.yaml`` with the resolved parameters."""
        yaml = YAML()
        yaml.default_flow_style = False

        data = {}
        if upstream:
            for up_section, up_params in upstream:
                if up_section in self.dynamic_key_sections:
                    resolved = dict(up_params or {})
                else:
                    resolved = dict(self.defaults[up_section])
                    for k, v in up_params.items():
                        if k in resolved and v is not None:
                            resolved[k] = v
                data[up_section] = resolved

        if section in self.dynamic_key_sections:
            resolved = dict(params or {})
        else:
            resolved = dict(self.defaults[section])
            for k, v in params.items():
                if k in resolved and v is not None:
                    resolved[k] = v
        data[section] = resolved

        out = directory / "config.yaml"
        with open(out, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh)
        return out
