# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-plot figure`` — drive many figures from one YAML spec.

A *figure spec* lists figures, each naming a **preset** (``scalar`` / ``profiles``
/ ``eps-chi`` — the existing subcommands) plus that subcommand's own options.
The driver resolves output directories from a perturb config (see
:mod:`odas_tpw.perturb.resolve`), compiles each figure entry into the
``argparse.Namespace`` the chosen ``run`` already accepts, and runs it — so
every plotting kernel and existing behaviour is reused, not reimplemented.

```yaml
source:                       # exactly one of config / root
  config: perturb.1.yaml
  # output_root: ~/Desktop/VMP_results   # optional override of where to look
  # root: ~/Desktop/VMP_results          # opt-out of config resolution

output_dir: figs/             # one subdir per figure is written here

sections:                     # optional; a file ref or an inline list
  file: sections.yaml

figures:
  - {name: ts,       preset: scalar,   vars: [JAC_T, SP, sigma0], depth_max: 150}
  - {name: overview, preset: eps-chi,  gap_seconds: 600}
  - {name: mixing,   preset: profiles, product: mixing, section: "*"}
```

A figure entry's keys are exactly the preset subcommand's options (``vars`` is
sugar for repeated ``--var``; ``clim`` is a ``{VAR: [min, max]}`` map). ``eps-chi``
has no x-axis, so ``section``/``vars``/``clim`` are rejected for it.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from odas_tpw.perturb.plot import eps_chi, profiles, scalar

# preset name -> (module providing add_arguments/run, default output kind)
_PRESETS: dict[str, Any] = {
    "scalar": scalar,
    "profiles": profiles,
    "eps-chi": eps_chi,
}
_PRESET_DIR = Path(__file__).parent / "presets"

# Friendly figure-spec keys that map to a differently-named Namespace attr.
_KEY_ALIAS = {"vars": "var"}
# Figure keys handled specially (not copied generically onto the Namespace).
_CONTROL_KEYS = {"name", "preset", "section"}
# Keys set by the source/CLI/driver — a figure may not override them.
_RESERVED_KEYS = {"config", "root", "strict", "latest", "out", "out_dir",
                  "sections", "select"}


class SpecError(SystemExit):
    """A user error in the figure spec (raised as a clean CLI exit)."""


# ---------------------------------------------------------------------------
# Spec loading / validation
# ---------------------------------------------------------------------------

def _load_spec(path: str) -> dict:
    try:
        with open(path) as fh:
            spec = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise SpecError(f"cannot read figure spec {path}: {exc}") from exc
    if not isinstance(spec, dict):
        raise SpecError(f"figure spec {path}: top level must be a mapping")
    return spec


def _validate_source(spec: dict) -> dict:
    source = spec.get("source")
    if not isinstance(source, dict):
        raise SpecError("figure spec: a 'source' mapping is required")
    has_config = bool(source.get("config"))
    has_root = bool(source.get("root"))
    if has_config == has_root:  # neither or both
        raise SpecError("figure spec: source needs exactly one of 'config' or 'root'")
    return source


def _sections_file(spec: dict, tmp: list[str]) -> str | None:
    """Resolve the spec's optional sections to a file path the subcommands read.

    Accepts ``sections: {file: PATH}`` or an inline ``sections: [ ... ]`` list
    (written to a temp file). Returns None when absent.
    """
    sections = spec.get("sections")
    if sections is None:
        return None
    if isinstance(sections, dict) and "file" in sections:
        return str(sections["file"])
    if isinstance(sections, list):
        fd, name = tempfile.mkstemp(suffix=".yaml", prefix="figspec_sections_")
        tmp.append(name)  # register before writing so a write error still cleans up
        with os.fdopen(fd, "w") as fh:
            yaml.safe_dump({"sections": sections}, fh)
        return name
    raise SpecError("figure spec: 'sections' must be {file: PATH} or a list")


# ---------------------------------------------------------------------------
# figure entry -> argparse.Namespace for the preset subcommand
# ---------------------------------------------------------------------------

def _adapt(key: str, value: Any) -> Any:
    """Convert a friendly spec value to the Namespace form the subcommand uses."""
    if key in ("vars", "var"):
        return list(value) if isinstance(value, (list, tuple)) else [value]
    if key == "clim":
        if not isinstance(value, dict):
            raise SpecError("figure 'clim' must be a {VAR: [min, max]} mapping")
        out = []
        for var, pair in value.items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise SpecError(f"figure 'clim' {var!r}: expected [min, max], got {pair!r}")
            out.append([str(var), str(pair[0]), str(pair[1])])
        return out
    return value


def _coerce(action: argparse.Action, key: str, value: Any, fig_name: str) -> Any:
    """Apply the parser action's ``type``/``choices`` to a YAML value, so a spec
    gets the same coercion/validation the CLI would (e.g. ``vmin: 1e-7`` parsed
    as a float, ``product: bogus`` rejected). Lists are coerced element-wise."""
    typ, choices = action.type, action.choices

    def one(v: Any) -> Any:
        # bool stays bool (store_true/false); don't float(True).
        if callable(typ) and v is not None and not isinstance(v, bool):
            try:
                v = typ(v)
            except (ValueError, TypeError) as exc:
                tname = getattr(typ, "__name__", str(typ))
                raise SpecError(
                    f"figure {fig_name!r}: option {key!r}={v!r} is not a valid {tname}"
                ) from exc
        if choices is not None and v not in choices:
            raise SpecError(
                f"figure {fig_name!r}: option {key!r}={v!r} must be one of "
                f"{sorted(str(c) for c in choices)}"
            )
        return v

    return [one(v) for v in value] if isinstance(value, list) else one(value)


def _apply_section(args: argparse.Namespace, preset: str, figure: dict, sections_file):
    section = figure.get("section")
    if preset == "eps-chi":
        if section is not None:
            raise SpecError(
                f"figure {figure.get('name')!r}: preset 'eps-chi' has no x-axis; "
                f"'section' is not allowed."
            )
        return
    if sections_file is not None:
        args.sections = sections_file
    if section is None or section == "*":
        args.select = None  # all sections
    elif isinstance(section, str):
        args.select = [section]
    elif isinstance(section, list):
        args.select = [str(s) for s in section]
    else:
        raise SpecError(f"figure {figure.get('name')!r}: bad 'section' {section!r}")
    if args.select and not sections_file:
        raise SpecError(
            f"figure {figure.get('name')!r}: 'section' names a section but the spec "
            f"has no 'sections' block."
        )


def _build_args(figure: dict, source: dict, sections_file, output_dir: Path,
                strict: bool, latest: bool) -> tuple[Any, argparse.Namespace]:
    preset = figure.get("preset")
    name = str(figure.get("name") or preset)
    if preset not in _PRESETS:
        raise SpecError(
            f"figure {name!r}: unknown preset {preset!r}; "
            f"choose from {sorted(_PRESETS)}"
        )
    mod = _PRESETS[preset]
    p = argparse.ArgumentParser(add_help=False)
    mod.add_arguments(p)
    args = p.parse_args([])
    dest_map = {a.dest: a for a in p._actions if a.dest and a.dest != "help"}

    # Source resolution (consumed by resolve.resolve_for_args inside run()).
    args.config = source.get("config")
    args.root = source.get("root") or source.get("output_root")
    args.strict = strict
    args.latest = latest

    # Generic per-preset options — coerced/validated by the subcommand's own
    # argparse action (type + choices), as if they had come from the CLI.
    for key, value in figure.items():
        if key in _CONTROL_KEYS:
            continue
        if key in _RESERVED_KEYS:
            raise SpecError(
                f"figure {name!r}: {key!r} is set by source/CLI, not per figure"
            )
        attr = _KEY_ALIAS.get(key, key)
        if attr not in dest_map:
            raise SpecError(
                f"figure {name!r}: option {key!r} is not valid for preset {preset!r}"
            )
        setattr(args, attr, _coerce(dest_map[attr], key, _adapt(key, value), name))

    _apply_section(args, preset, figure, sections_file)

    # Output: one subdir per figure so different figures never collide.
    fig_dir = output_dir / _safe(name)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if preset == "eps-chi":
        args.out = str(fig_dir / "eps_chi.png")
    else:
        args.out_dir = str(fig_dir)
    return mod.run, args


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the figure subcommand on *p*."""
    p.add_argument("--spec", help="figure-spec YAML (layout + contents).")
    p.add_argument("--select", action="append", default=None, metavar="NAME",
                   help="render only the named figure(s) (repeatable). Default: all.")
    p.add_argument("--strict", action="store_true",
                   help="error (don't fall back) when a config signature doesn't match.")
    p.add_argument("--latest", action="store_true",
                   help="use the newest {stage}_NN even if no signature matches.")
    p.add_argument("--list-presets", action="store_true",
                   help="list the bundled example presets and exit.")
    p.add_argument("--dump-preset", metavar="NAME",
                   help="print the bundled example spec for a preset and exit.")


def _list_presets() -> None:
    print("Bundled example figure specs (copy and edit):")
    for f in sorted(_PRESET_DIR.glob("*.yaml")):
        print(f"  {f.stem}")
    print("\nDump one with:  perturb-plot figure --dump-preset NAME")


def _dump_preset(name: str) -> None:
    path = _PRESET_DIR / f"{name}.yaml"
    if not path.exists():
        avail = ", ".join(sorted(p.stem for p in _PRESET_DIR.glob("*.yaml")))
        raise SpecError(f"no example preset {name!r}; available: {avail}")
    print(path.read_text())


def run(args: argparse.Namespace) -> str:
    """Render every figure in the spec. Returns the output directory."""
    if args.list_presets:
        _list_presets()
        return ""
    if args.dump_preset:
        _dump_preset(args.dump_preset)
        return ""
    if not args.spec:
        raise SpecError("figure: --spec is required (or use --list-presets/--dump-preset)")

    spec = _load_spec(args.spec)
    source = _validate_source(spec)
    figures = spec.get("figures")
    if not isinstance(figures, list) or not figures:
        raise SpecError("figure spec: a non-empty 'figures' list is required")
    output_dir = Path(spec.get("output_dir", ".")).expanduser()

    # Up-front: every entry is a mapping, and no two figures collide on the
    # output subdir name (post-sanitization).
    names: list[Any] = []
    safe_seen: dict[str, Any] = {}
    for f in figures:
        if not isinstance(f, dict):
            raise SpecError("figure spec: each figures[] entry must be a mapping")
        nm = f.get("name") or f.get("preset")
        names.append(nm)
        sn = _safe(str(nm))
        if sn in safe_seen:
            raise SpecError(
                f"figure spec: figures {safe_seen[sn]!r} and {nm!r} map to the "
                f"same output name {sn!r}"
            )
        safe_seen[sn] = nm

    wanted = set(args.select) if args.select else None
    if wanted:
        missing = wanted - set(names)
        if missing:
            raise SpecError(f"figure --select: unknown figure name(s) {sorted(missing)}")

    tmp: list[str] = []
    try:
        sections_file = _sections_file(spec, tmp)
        rendered = 0
        for figure in figures:
            name = figure.get("name") or figure.get("preset")
            if wanted and name not in wanted:
                continue
            run_fn, fig_args = _build_args(
                figure, source, sections_file, output_dir, args.strict, args.latest
            )
            print(f"=== figure {name!r} (preset {figure.get('preset')!r}) ===")
            run_fn(fig_args)
            rendered += 1
        print(f"Rendered {rendered} figure(s) into {output_dir}")
    finally:
        for t in tmp:
            with contextlib.suppress(OSError):
                os.unlink(t)
    return str(output_dir)
