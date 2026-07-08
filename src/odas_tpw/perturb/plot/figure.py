# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-plot figure`` — drive many figures from one YAML spec.

A *figure spec* lists figures, each naming a **preset** (``scalar`` / ``profiles``
/ ``epsilon`` / ``chi`` / ``mixing`` / ``eps-chi`` — the ``perturb-plot``
subcommands) plus that subcommand's own options. The driver resolves output
directories from a perturb config (see :mod:`odas_tpw.perturb.resolve`),
compiles each figure entry into the ``argparse.Namespace`` the chosen ``run``
already accepts, and runs it — so every plotting kernel and existing behavior
is reused, not reimplemented.

```yaml
source:                       # exactly one of config / root
  config: perturb.1.yaml
  # output_root: ~/Desktop/VMP_results   # optional override of where to look
  # root: ~/Desktop/VMP_results          # opt-out of config resolution

output_dir: figs/             # one subdir per figure is written here

sections:                     # optional; a file ref or an inline list
  file: sections.yaml

figures:
  - {name: ts,       preset: scalar,  vars: [JAC_T, SP, sigma0], depth_max: 150}
  - {name: overview, preset: eps-chi, gap_seconds: 600}
  - {name: mixing,   preset: mixing,  section: "*"}
```

A figure entry's keys are exactly the preset subcommand's options (``vars`` is
sugar for repeated ``--var``; ``clim`` is a ``{VAR: [min, max]}`` map). ``eps-chi``
has no x-axis, so ``section``/``vars``/``clim`` are rejected for it.

With no output destination (neither the spec nor the CLI sets ``output_dir`` /
``output_pdf``) the figures are shown on screen when a display is available,
falling back to a PNG tree in the cwd when it is not (headless / no tty / a
non-GUI backend) — mirroring the ``scalar``/``profiles`` subcommands. Set
``output_dir`` or ``output_pdf`` to force writing files.

The spec's ``source`` (``config``/``root``), output destination
(``output_dir``/``output_pdf``), and ``sections:`` block can be overridden from
the command line — e.g. ``perturb-plot figure --spec fig.yaml --config
perturb.2.yaml --output-dir figs/run2`` — so one spec renders the same figures
across several perturb runs for comparison without editing the file.

Two independent subsetting flags (note they select different things):

* ``--figure NAME`` (repeatable) renders only the named *figures* from the
  spec's ``figures:`` list.
* ``--sections SECTIONS.YAML`` replaces the spec's ``sections:`` block, and
  ``--select NAME`` (repeatable, comma-separated) renders only the named
  *sections* — narrowing every figure's own ``section:`` choice, exactly as
  ``perturb-plot scalar --select`` does. ``eps-chi`` has no x-axis, so it
  ignores ``--select``; a figure narrowed to no sections is skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from odas_tpw.perturb.plot import eps_chi, overview, profiles, scalar, sections

# preset name -> object providing add_arguments/build_figures/run. The binned
# (bin, profile) presets are ProductView instances (one product each); scalar
# and eps-chi are plain modules. All duck-type the same surface.
_PRESETS: dict[str, Any] = {
    "scalar": scalar,
    "profiles": profiles.PROFILES,
    "epsilon": profiles.EPSILON,
    "chi": profiles.CHI,
    "mixing": profiles.MIXING,
    "eps-chi": eps_chi,
    "overview": overview,
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
            spec = YAML(typ="safe").load(fh)
    except (OSError, YAMLError) as exc:
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


def _apply_cli_overrides(spec: dict, args: argparse.Namespace) -> None:
    """Let CLI flags override the spec's source / output destination, in place.

    Enables comparing perturb runs from a single spec: ``--config`` / ``--root``
    swap where directories are resolved from, and ``--output-dir`` /
    ``--output-pdf`` redirect where the figures land — all without editing the
    spec file. Validation of the resulting values is left to
    :func:`_validate_source` / :func:`_output_config`.
    """
    cli_config = getattr(args, "config", None)
    cli_root = getattr(args, "root", None)
    if cli_config and cli_root:
        raise SpecError("figure: --config and --root are mutually exclusive")
    if cli_config or cli_root:
        src = spec.get("source")
        # Keep a spec source's output_root (a search-location override) when only
        # the config/root selector is swapped; drop whichever selector we replace.
        src = dict(src) if isinstance(src, dict) else {}
        src.pop("config", None)
        src.pop("root", None)
        if cli_root:
            src.pop("output_root", None)  # a raw root is itself the search dir
            src["root"] = cli_root
        else:
            src["config"] = cli_config
        spec["source"] = src

    cli_dir = getattr(args, "output_dir", None)
    cli_pdf = getattr(args, "output_pdf", None)
    if cli_dir and cli_pdf:
        raise SpecError("figure: --output-dir and --output-pdf are mutually exclusive")
    if cli_dir:
        spec["output_dir"] = cli_dir
        spec.pop("output_pdf", None)
    elif cli_pdf:
        spec["output_pdf"] = cli_pdf
        spec.pop("output_dir", None)

    # --sections replaces the spec's whole sections block (like --config swaps
    # the source), so one spec can be re-sectioned from the CLI without editing.
    cli_sections = getattr(args, "sections", None)
    if cli_sections:
        spec["sections"] = {"file": cli_sections}


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
            YAML(typ="safe").dump({"sections": sections}, fh)
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
    """Apply the parser action's coercion/validation to a YAML value, so a spec
    gets the same handling the CLI would: ``vmin: 1e-7`` floats, ``product:
    bogus`` is rejected, ``apply_qc: "false"`` is rejected (truthy string), a
    list for a scalar option is rejected, and an nargs option (e.g. ``point:``
    for ``--point nargs=2``) must be a list of the declared length."""
    # store_true/false: the value is the destination boolean directly. A bare
    # string ("false") would be truthy downstream, so require a real bool.
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            raise SpecError(
                f"figure {fig_name!r}: option {key!r} expects true/false, got {value!r}"
            )
        return value

    # A list is only valid for list-producing actions (append, or nargs +/*/N).
    takes_list = (
        isinstance(action, argparse._AppendAction)
        or action.nargs in ("*", "+")
        or isinstance(action.nargs, int)
    )
    if isinstance(value, list) and not takes_list:
        raise SpecError(
            f"figure {fig_name!r}: option {key!r} takes a single value, got list {value!r}"
        )

    # An nargs option (e.g. --point nargs=2) consumes a fixed/variable number of
    # values in ONE go — so the spec value must be a list of the right length,
    # exactly as the CLI enforces. (append is different: a list there means
    # several invocations, each consuming nargs values, so it isn't length-
    # checked here — its per-invocation shape is handled by `_adapt`.)
    nargs_list = takes_list and not isinstance(action, argparse._AppendAction)
    if nargs_list and not isinstance(value, list):
        raise SpecError(
            f"figure {fig_name!r}: option {key!r} expects a list of values, "
            f"got {value!r}"
        )
    if nargs_list and isinstance(value, list):
        if isinstance(action.nargs, int) and len(value) != action.nargs:
            raise SpecError(
                f"figure {fig_name!r}: option {key!r} expects exactly "
                f"{action.nargs} values, got {len(value)}: {value!r}"
            )
        if action.nargs == "+" and not value:
            raise SpecError(
                f"figure {fig_name!r}: option {key!r} requires at least one value"
            )

    typ, choices = action.type, action.choices

    def one(v: Any) -> Any:
        # A YAML bool reaching here is for a typed/positional option (store_true/
        # store_false were handled above), never valid — reject it BEFORE
        # coercion so `dpi: true` / `figsize: [true, 9]` / `depth_max: true` fail
        # with a clean SpecError instead of crashing matplotlib at render time.
        if isinstance(v, bool):
            raise SpecError(
                f"figure {fig_name!r}: option {key!r}={v!r} is a boolean; "
                f"expected {getattr(typ, '__name__', 'a value')}"
            )
        # str() first so coercion matches argparse exactly (int('1.5') raises —
        # no silent float->int truncation).
        if callable(typ) and v is not None:
            try:
                v = typ(str(v))
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


def _apply_section(args: argparse.Namespace, preset: str, figure: dict, sections_file,
                   section_select: frozenset[str] | None = None) -> bool:
    """Set the leaf's section selection from the figure's own ``section:`` key,
    narrowed by the global ``--select`` (*section_select*: a validated set of
    section names, or None when ``--select`` was not given).

    Returns True when the figure still has sections to render, False when the
    global ``--select`` leaves this figure with none (the caller then skips it).
    ``eps-chi`` has no x-axis, so sections do not apply: the global ``--select``
    is a no-op and a per-figure ``section:`` is rejected.
    """
    section = figure.get("section")
    if preset == "eps-chi":
        if section is not None:
            raise SpecError(
                f"figure {figure.get('name')!r}: preset 'eps-chi' has no x-axis; "
                f"'section' is not allowed."
            )
        return True  # global --select does not apply to a section-less preset
    if sections_file is not None:
        args.sections = sections_file
    # The figure's own choice: None / "*" mean "every section in the file".
    if section is None or section == "*":
        own: list[str] | None = None
    elif isinstance(section, str):
        own = [section]
    elif isinstance(section, list):
        own = [str(s) for s in section]
    else:
        raise SpecError(f"figure {figure.get('name')!r}: bad 'section' {section!r}")
    # Narrow by the global --select. When the figure takes "all", the global set
    # IS the selection; otherwise keep only the figure's sections that are in it.
    if section_select is None:
        effective = own
    elif own is None:
        effective = sorted(section_select)
    else:
        effective = [s for s in own if s in section_select]
        if not effective:
            return False  # nothing survives the --select narrowing; skip figure
    args.select = effective  # None = every section
    if args.select and not sections_file:
        raise SpecError(
            f"figure {figure.get('name')!r}: 'section' names a section but the spec "
            f"has no 'sections' block."
        )
    return True


def _build_args(figure: dict, source: dict, sections_file, output_dir: Path,
                strict: bool, latest: bool, section_select: frozenset[str] | None = None,
                *, make_output: bool = True) -> tuple[Any, argparse.Namespace] | None:
    """Compile a figure entry into ``(preset_module, Namespace)``.

    Returns ``None`` when the global ``--select`` (*section_select*) narrows this
    figure to no sections — the caller skips it (no output dir is created).

    With ``make_output`` (PNG mode) the per-figure output path is assigned and
    its subdir created; the PDF driver passes ``make_output=False`` because it
    streams ``build_figures()`` into one document and needs no subdirs.
    """
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
        # Accept the CLI spelling (hyphens) as well as the dest (underscores).
        attr = _KEY_ALIAS.get(key, key).replace("-", "_")
        if attr in _RESERVED_KEYS:
            raise SpecError(
                f"figure {name!r}: {key!r} is set by source/CLI, not per figure"
            )
        if attr not in dest_map:
            valid = sorted(k for k in dest_map if k not in _RESERVED_KEYS)
            raise SpecError(
                f"figure {name!r}: option {key!r} is not valid for preset "
                f"{preset!r}. Valid options: {valid}"
            )
        setattr(args, attr, _coerce(dest_map[attr], key, _adapt(key, value), name))

    if not _apply_section(args, preset, figure, sections_file, section_select):
        return None  # --select narrowed this figure to no sections; caller skips

    if make_output:
        # PNG mode: one subdir per figure so different figures never collide.
        fig_dir = output_dir / _safe(name)
        fig_dir.mkdir(parents=True, exist_ok=True)
        if preset == "eps-chi":
            args.out = str(fig_dir / "eps_chi.png")
        else:
            args.out_dir = str(fig_dir)
    return mod, args


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the figure subcommand on *p*."""
    p.add_argument("--spec", help="figure-spec YAML (layout + contents).")
    p.add_argument("--figure", action="append", default=None, metavar="NAME",
                   help="render only the named figure(s) from the spec's 'figures:' "
                        "list (repeatable). Default: all.")
    p.add_argument("--sections", default=None, metavar="SECTIONS.YAML",
                   help="override the spec's 'sections:' block with this sections "
                        "YAML (same file 'perturb-plot scalar --sections' reads).")
    p.add_argument("--select", action="append", default=None, metavar="NAME",
                   help="render only the named section(s) (repeatable, or "
                        "comma-separated), narrowing every figure's own section "
                        "choice. Default: every section. Requires sections.")
    p.add_argument("--config", metavar="PERTURB.YAML", default=None,
                   help="override the spec's source config (compare/contrast "
                        "perturb runs from one spec). Mutually exclusive with --root.")
    p.add_argument("--root", metavar="DIR", default=None,
                   help="override the spec's source: resolve dirs by newest "
                        "{stage}_NN under DIR instead of a config. "
                        "Mutually exclusive with --config.")
    p.add_argument("--output-dir", metavar="DIR", default=None,
                   help="override the spec's output_dir (one PNG tree). "
                        "Mutually exclusive with --output-pdf.")
    p.add_argument("--output-pdf", metavar="PDF", default=None,
                   help="override the spec's output_pdf (one combined PDF). "
                        "Mutually exclusive with --output-dir.")
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


def _output_config(spec: dict) -> tuple[str | None, str | None, int | None]:
    """Validate the spec's output controls.

    At most one of ``output_dir`` (one PNG tree) or ``output_pdf`` (one combined
    multipage PDF); plus an optional default ``dpi`` applied to any figure that
    doesn't set its own. Both may be ``None`` -- with no output destination the
    figures are displayed on screen when a display is available, else written to
    a PNG tree in the cwd (decided in :func:`run`).
    """
    out_dir = spec.get("output_dir")
    out_pdf = spec.get("output_pdf")
    if out_dir is not None and out_pdf is not None:
        raise SpecError("figure spec: set only one of 'output_dir' or 'output_pdf'")
    dpi = spec.get("dpi")
    if dpi is not None:
        if isinstance(dpi, bool) or not isinstance(dpi, int):
            raise SpecError(f"figure spec: 'dpi' must be an integer, got {dpi!r}")
        if dpi <= 0:
            raise SpecError(f"figure spec: 'dpi' must be a positive integer, got {dpi}")
    return out_dir, out_pdf, dpi


def _compiled_figures(figures, source, sections_file, args, wanted, default_dpi,
                      output_dir: Path, section_select, *, make_output: bool):
    """Yield ``(name, preset_module, Namespace)`` for each selected figure, with
    the spec's default ``dpi`` filled in where the figure didn't set its own.

    A figure the global ``--select`` narrows to no sections is skipped (a note is
    printed) rather than rendered empty."""
    for figure in figures:
        name = figure.get("name") or figure.get("preset")
        if wanted and name not in wanted:
            continue
        built = _build_args(
            figure, source, sections_file, output_dir, args.strict, args.latest,
            section_select, make_output=make_output,
        )
        if built is None:
            print(f"figure {name!r}: no sections remain after --select; skipped")
            continue
        mod, fig_args = built
        if getattr(fig_args, "dpi", None) is None:
            fig_args.dpi = default_dpi
        print(f"=== figure {name!r} (preset {figure.get('preset')!r}) ===")
        yield name, mod, fig_args


def _resolve_section_select(select, sections_file) -> frozenset[str] | None:
    """Validate the global ``--select`` names against the sections file and return
    them as a set (or None when ``--select`` was not given).

    ``--select`` needs sections to select from — either the spec's ``sections:``
    block or a ``--sections`` override. Unknown names are rejected here (via
    ``sections.select_sections``) so the error names them, rather than silently
    narrowing every figure to nothing.
    """
    if not select:
        return None
    if sections_file is None:
        raise SpecError(
            "figure --select requires sections: add a 'sections:' block to the "
            "spec or pass --sections SECTIONS.YAML"
        )
    available = sections.load_sections(sections_file)
    chosen = sections.select_sections(available, select)  # raises on unknown name
    return frozenset(s.name for s in chosen)


def run(args: argparse.Namespace) -> str:
    """Render every figure in the spec. Returns the PNG dir or the PDF path."""
    if args.list_presets:
        _list_presets()
        return ""
    if args.dump_preset:
        _dump_preset(args.dump_preset)
        return ""
    if not args.spec:
        raise SpecError("figure: --spec is required (or use --list-presets/--dump-preset)")

    spec = _load_spec(args.spec)
    _apply_cli_overrides(spec, args)  # --config/--root/--output-* win over the spec
    source = _validate_source(spec)
    figures = spec.get("figures")
    if not isinstance(figures, list) or not figures:
        raise SpecError("figure spec: a non-empty 'figures' list is required")
    out_dir, out_pdf, default_dpi = _output_config(spec)

    # Up-front: every entry is a mapping, and no two figures collide on their
    # name (the PNG subdir / the --select key), post-sanitization.
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
                f"figure spec: figures {safe_seen[sn]!r} and {nm!r} both map to "
                f"the same name {sn!r} — give each figure a unique 'name:'."
            )
        safe_seen[sn] = nm

    wanted = set(args.figure) if args.figure else None
    if wanted:
        missing = wanted - set(names)
        if missing:
            raise SpecError(f"figure --figure: unknown figure name(s) {sorted(missing)}")

    tmp: list[str] = []
    try:
        sections_file = _sections_file(spec, tmp)
        section_select = _resolve_section_select(args.select, sections_file)
        if out_dir is None and out_pdf is None and sections.can_display():
            # No output destination + a display available: show on screen,
            # mirroring the scalar/profiles subcommands. Headless/CI (no tty or a
            # non-GUI backend) falls through to a PNG tree in the cwd below.
            result = _render_show(
                figures, source, sections_file, args, wanted, default_dpi,
                section_select,
            )
        elif out_pdf is not None:
            result = _render_pdf(
                figures, source, sections_file, args, wanted, default_dpi, out_pdf,
                section_select,
            )
        else:
            output_dir = Path(out_dir or ".").expanduser()
            rendered = 0
            for _name, mod, fig_args in _compiled_figures(
                figures, source, sections_file, args, wanted, default_dpi,
                output_dir, section_select, make_output=True,
            ):
                mod.run(fig_args)
                rendered += 1
            print(f"Rendered {rendered} figure(s) into {output_dir}")
            result = str(output_dir)
    finally:
        for t in tmp:
            with contextlib.suppress(OSError):
                os.unlink(t)
    return result


def _render_show(figures, source, sections_file, args, wanted, default_dpi,
                 section_select=None) -> str:
    """Display the selected figures on screen (blocking), writing no files.

    Streams each figure's ``build_figures`` (``make_output=False``, so no PNG
    subdirs) and hands the open figures to ``sections.save_or_show`` with no
    output dir, which holds them all open and calls ``plt.show()``. ``run`` only
    routes here when ``sections.can_display()`` is true, so an interactive
    backend is assumed. Note a broad spec (many figures x many sections) opens
    that many windows at once.
    """
    def _stream():
        for _name, mod, fig_args in _compiled_figures(
            figures, source, sections_file, args, wanted, default_dpi,
            Path("."), section_select, make_output=False,
        ):
            # closing_figs releases each preset's dataset handle; close_new_figs
            # closes a figure orphaned by a build error before it was yielded.
            with sections.closing_figs(mod.build_figures(fig_args)) as gen, \
                    sections.close_new_figs_on_error():
                yield from gen

    shown = sections.save_or_show(_stream(), None, default_dpi or 150)
    return f"displayed {shown} figure(s)"


def _render_pdf(figures, source, sections_file, args, wanted, default_dpi,
                out_pdf: str, section_select=None) -> str:
    """Render every selected figure into one multipage PDF (one page per figure
    the preset produces).

    Written to a temp file and ``os.replace``-d into place only on FULL success,
    so a failure in a later figure never leaves a partial PDF that looks
    complete but is silently missing pages; an all-empty spec raises and leaves
    nothing. Figures stream one at a time and are closed as written.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = Path(out_pdf).expanduser()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".pdf", prefix=".figspec_",
                                    dir=str(pdf_path.parent))
    os.close(fd)
    pages = 0
    try:
        with PdfPages(tmp_name) as pdf:
            for _name, mod, fig_args in _compiled_figures(
                figures, source, sections_file, args, wanted, default_dpi,
                Path("."), section_select, make_output=False,
            ):
                # closing_figs releases the preset's dataset handle even if
                # savefig raises mid-stream; close_new_figs_on_error closes a
                # figure orphaned by a build error before it was yielded (e.g.
                # eps-chi, which builds its figure inline).
                with sections.closing_figs(mod.build_figures(fig_args)) as gen, \
                        sections.close_new_figs_on_error():
                    for _stem, fig in gen:
                        try:
                            pdf.savefig(fig, dpi=fig_args.dpi or 150)
                            pages += 1
                        finally:
                            plt.close(fig)
            if pages == 0:
                raise SpecError(
                    "figure spec: no figures were produced — "
                    "nothing to write to the PDF"
                )
        os.replace(tmp_name, str(pdf_path))  # atomic; only after full success
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    print(f"Wrote {pages} page(s) to {pdf_path}")
    return str(pdf_path)
