# Generalized YAML-driven plotting for `perturb-plot` — design plan (v2, post-review)

> v1 proposed a generic per-panel renderer with a variable registry. Two adversarial
> reviews (feasibility + scope) found a blocker and several majors and converged on a
> simpler, lower-risk design. This v2 is that design. v1 findings are summarized in §9.

## 1. What the user asked for, decomposed

1. **Point at the perturb config YAML → right directories are used.** → component **A (resolver)**.
2. **Predetermined sets "like scalar".** → the *existing* `scalar`/`profiles`/`eps-chi` figures.
3. **One YAML for plot layout + contents.** → component **B (batch spec)** listing figures,
   each a preset + that figure's options + output/size controls.

## 2. Design in one line

A YAML **batch-driver**: each figure entry = `{preset, <that subcommand's options>}`. The
driver resolves directories from the perturb config (A), compiles each entry to the
`argparse.Namespace` the existing `scalar.run`/`profiles.run`/`eps_chi.run` already accept,
and runs them — collecting the figures to PNGs or one multipage PDF. **No new renderer, no
variable registry, no mixed-product panel grid.** The existing functions remain the engines,
so every plotting kernel and every `test_perturb_plot_*` test is reused unchanged.

## 3. Component A — read-only config→directory resolver

The blocker (review F-1): the stage hash includes `files.output_root`/`p_file_root`
(`config.py:21-28`, in every chain via `pipeline.py:184`), and `--output`/`--p-file-root`
overwrite those at run time **without normalization** (`cli.py:87-88,132-135`). So
recomputing the hash from the YAML alone will mismatch whenever `--output` was used → silent
miss. Therefore **do not recompute the hash**.

Instead, **compare against the stored signature**: each `{stage}_NN` dir holds
`.params_sha256_{hash}` whose *body* is the canonical JSON of that stage's params + upstream
chain (`config_base.py:309-321`). The resolver:

```python
# src/odas_tpw/perturb/resolve.py  (new, public, read-only)
def stage_dir(config: dict, stage: str, output_root: str|Path|None=None) -> Path:
    """Find the existing {prefix}_NN dir whose signature matches `config`,
    comparing canonical params section-by-section but IGNORING `files`
    (run-environment: output_root/p_file_root/p_file_pattern). Raises a
    descriptive FileNotFoundError if none/ambiguous. Never creates a dir."""
```

Mechanism: merge `config` exactly as the pipeline does (`merge_config`), build the stage's
canonical params via the **same** code the pipeline uses, read each candidate dir's
`.params_sha256_*` body, JSON-parse, drop the `files` section from both sides, and compare.
Two shared-code requirements (reviews S2-m10/m11):

```python
# Relocate _upstream_for + its stage→chain table from pipeline.py into perturb/config.py
# (pure config logic; pipeline imports it from there). Add the (section, params, upstream)
# triple builder there too, called by BOTH _run_combo and resolve.py:
def stage_signature(stage: str, config: dict) -> tuple[str, dict, list[tuple[str,dict]]]:
    """The SAME triple the pipeline passes to resolve_output_dir.
    Combos use section='binning'/'ctd' (review F-2), not the source stage."""

# Factor a READ-ONLY finder out of resolve_output_dir (don't re-glob — review m10):
def find_output_dir(base, prefix, hash) -> Path | None: ...   # no creation
```

This kills review F-1 (path differences live only in `files`, which we ignore), F-2 (combos
hash under `section='binning'`/`'ctd'` — now in one place), m10 (no forked glob), and m11
(single config-logic home). Stage→prefix: `profiles→combo`, `diss→diss_combo`,
`chi→chi_combo`, `ctd→ctd_combo`.

**Failure UX — the two scope reviews disagree; reconciled middle (reviews S-4 vs S2-M2):**
S-4 wants hard-error (never plot the wrong config); S2-M2 wants warn+fallback (the data is
right there). Resolve by *cardinality*, since the real hazard is only ambiguity:
- exact signature match → use it.
- no match **and exactly one** `{prefix}_NN` exists → use it with a **loud warning** naming
  the drift (the common "tweak config, re-plot" case; can't be the wrong config — there's
  only one).
- no match **and ≥2** dirs exist → **hard-error**, listing each dir + its stored config, since
  guessing could plot a *different* config's data (the exact bug P1 fixes).
- `--strict` forces error on any miss; `--latest` forces newest. Disabled-stage / never-run
  are classified distinctly in the message. Each product resolves independently, so report
  per-product (review n15: `eps-chi` reads `diss_combo` + `chi_combo`, possibly different
  `_NN` — resolve each).

`A` is wired into the existing subcommands too: add `--config PATH` to `scalar`/`profiles`/
`eps-chi`; when given, directory discovery uses `resolve.stage_dir` instead of `--root` +
`latest_stage_dir`. This alone delivers ask #1 for CLI users.

## 4. Component B — the batch spec YAML

```yaml
source:                              # exactly one of {config, root}
  config: perturb.1.yaml             # resolve dirs by hash (A); output_root from it
  # output_root: ~/Desktop/VMP_results   # optional override of WHERE to look
  # root: ~/Desktop/VMP_results          # opt-out: latest_NN (no config)

output:
  dir: figs/                         # one PNG per figure  (XOR pdf:)
  # pdf: report.pdf                  # one multipage PDF instead
  dpi: 150

sections: {file: sections.yaml}      # OR an inline list (review S-7: reference allowed)

figures:
  - preset: eps-chi                  # opaque preset -> eps_chi.run; NO section/x-axis
    title: "ε / χ overview"
    gap_seconds: 600

  - preset: scalar                   # -> scalar.run
    title: "T/S along NE line"
    section: nleg                    # name lookup into `sections`
    vars: [JAC_T, SP, sigma0]
    depth_max: 150
    clim: {JAC_T: [18, 28]}
    figsize: [11, 9]

  - preset: profiles                 # -> profiles.run
    product: chi
    vars: [chiMean, K_rho]
    section: full
    p_max: 150
```

- **One inheritance edge only** (review S-1): figure-level fields override `defaults`; there is
  **no** `overrides:`/preset-patch layer. To customize a preset, the user copies the preset's
  example YAML and edits the figure entry. `vars:` is plain sugar that the *existing*
  subcommands already accept (repeatable `--var`), so no new semantics.
- A figure entry's keys are exactly the chosen subcommand's existing CLI options. The driver
  builds the `Namespace` and calls `run()`. Validation = "is this a known option of that
  preset?" plus reusing `sections.validate_params` for x-axis blocks.
- **`section:` accepts a name, a list, or `"*"`** (review S2-m7): `section: "*"` fans the
  figure out to one output per section — preserving today's "render product X for *all* my
  sections in one invocation" workflow (the whole point of `sections.yaml`), which a
  one-section-per-figure model would otherwise regress.
- **eps-chi is the one Python-only preset** (review F-3/S-5): it renders depth-vs-cast-number
  with no x-axis, so `section:`/`xaxis:` are rejected for it (clear error), not silently
  ignored. Examples never show `section:` on eps-chi.

## 5. Component C — presets as shippable YAML (review S-6)

Ship `src/odas_tpw/perturb/plot/presets/{scalar,profiles}.yaml` — the *same* batch-spec
schema, consumed by the *same* driver, so a preset and a hand-written figure can't drift.
`perturb-plot figure --list-presets` and `--dump-preset scalar` print them for copy-edit.
`eps-chi` is documented as the lone non-copyable preset (Python).

## 6. Output controls the word "layout" implies (review S-3)

Per-figure `title`, `figsize`, and top-level `output.dpi`/`format`, plus **one combined
multipage `pdf:`** vs per-figure PNGs. (Within-figure panel arrangement stays the preset's —
`scalar`/`profiles` are N×1, `eps-chi` is 3×1.) Cross-panel shared colorbars and gridspec
spans remain non-goals.

## 7. Explicit non-goals (scope fence)

- **No generic mixed-product panel grid** (one figure mixing `diss`+`chi`+`ctd` panels). This
  was v1's core and the source of every feasibility blocker: the flat var→scale registry
  breaks `N2` (linear in scalar, symlog in profiles — review F-4), the χ/ε ratio is a
  two-combo cross-regrid not a "synthetic product" (F-3), and depth(m)-vs-pressure(dbar)
  y-axes can't share an axis (S-3). Deferred until a real figure actually needs it; when it
  does, it's an additive new preset/renderer, not a prerequisite.
- No expression DSL, no gridspec, no map panels.

## 8. CLI surface

- `perturb-plot figure --spec figure.yaml [--select NAME] [--latest]` — the batch driver.
- `--config PATH` added to `scalar`/`profiles`/`eps-chi` (single-figure, config-resolved).
- `--list-presets` / `--dump-preset NAME`.

## 9. v1 review findings folded in

- **F-1 (blocker):** hashed `files.*` rewritten by `--output` → resolve by stored-signature
  comparison excluding `files`, not by recomputing the hash. (§3)
- **F-2:** combos hash under `section='binning'`/`'ctd'` → shared `stage_signature` helper. (§3)
- **F-3:** eps_chi is a third render path; χ/ε ratio is two-source → eps-chi stays an opaque
  preset, section rejected. (§4)
- **F-4:** flat var→scale registry breaks N2 → **registry dropped entirely** (no generic
  renderer). (§7)
- **F-5/F-7:** `sections.resolve_sections`/`load_sections` aren't dict-reusable; hashing is a
  `ConfigManager` method re-exported via `perturb.config` → resolver calls
  `perturb.config.*` + the shared `stage_signature`; spec reuses only `_section_from_dict`/
  `validate_params`.
- **F-6:** legacy "thin wrappers" leak (pseudo-vars, per-quantity flags, interactive display)
  → legacy `run()` bodies kept **as the engines**, not reimplemented. (§2)
- **S-1:** drop the three-layer content model → figure-level options + explicit list. (§4)
- **S-2/verdict:** scope too big → batch-driver, defer generic renderer. (§2, §7)
- **S-3:** mixed depth/pressure sharey wrong + missing figsize/dpi/pdf → one product per
  figure; add output controls. (§6, §7)
- **S-4:** config-drift UX → hard-error with stage/classification/listing + `--latest`. (§3)
- **S-6:** presets as copyable YAML, not Python. (§5)
- **S-7:** reference external sections file; reconcile "not the config" docs. (§4)

Second scope review (the slow one) — net-new, folded in:
- **S2-M1/M2:** combo-hash inputs + failure UX → §3 (cardinality-based fallback).
- **S2-M3 (end-state):** the `figure` spec is the **primary** interface going forward; the
  three legacy subcommands remain (they are the engines, and still usable directly) but docs
  lead with `figure`, and the legacy flag-pile is frozen, not grown. Defer `dir:` panel
  escape hatch (n13) and trim `source:` to "exactly one of {config, root}" (n14).
- **S2-m7:** `section: "*"`/list expansion (§4). **S2-m10/m11:** shared `find_output_dir` +
  config-logic home (§3).
- **S2 "missing capability":** a **shared colour scale across like-variable figures** (two ε
  figures, or ε across two configs, visually comparable). Noted as a P2 stretch — cheap when
  the driver controls each figure's `clim`; a `clim_group:` could pin a common scale. Kept
  out of the hard scope for now; revisit after P2 lands.

## 10. Phasing

- **P1 — resolver (A):** `resolve.stage_dir` + shared `stage_signature`; `--config` on the
  three subcommands. Delivers ask #1. Self-contained, testable, no UI churn.
- **P2 — batch driver (B+C):** `figure` subcommand, spec loader/validator, preset YAMLs,
  PNG/PDF output. Delivers asks #2/#3. Built entirely on P1 + existing `run()`s.
- **P3 — (optional, future):** generic mixed-product figure, only if a concrete need appears.

## 11. Testing

- Resolver: run the pipeline, then resolve every stage dir for the same config (incl. a run
  invoked with `--output` to a non-config path — must still resolve); disabled-stage and
  config-drift produce the classified error; multi-config root disambiguation.
- Driver: spec→Namespace compilation per preset; unknown-option and eps-chi-with-section
  errors; `--select`; PNG vs PDF output; preset round-trip (`--dump-preset` re-parses).
- Legacy: existing `test_perturb_plot_*` unchanged (the engines are untouched); add a
  `--config`-resolves-correct-dir case.
