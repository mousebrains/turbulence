# Rationalizing the AEM1-G split

The Bank Seaspider EM calibration work was split out of this repo into
`~/tpw/AEM1-G` (`github.com/mousebrains/AEM1-G`) on 2026-08-26. The split was
done by copying forward what the paper needed; nothing was deleted here
afterwards. This document says where each surviving item belongs and why, so
the cleanup is a decision made once rather than re-litigated per file.

Written 2026-09-06 against `turbulence@c1853fa` and `AEM1-G@5574612`.

---

## 1. The rule

> **turbulence owns instrument and format knowledge that outlives any one
> deployment. AEM1-G owns one dataset, its results and its manuscript.**

The operational test: *would the next deployment need this, if the AEM1-G paper
were abandoned tomorrow?* If yes it belongs here, however much it was written
for the paper. If no it belongs there, however general it looks.

A corollary that decides several cases below: **a script that has to import
both packages belongs in AEM1-G**, because AEM1-G already declares a dependency
on `odas_tpw` and this repo declares none on `aem1g`. The dependency arrow
points one way and must keep doing so.

---

## 2. `clocksync` stays here — the arguable case, argued

The instinct to move it is understandable: it was written for the Bank
Seaspider mooring, it was used exactly once, and the answer it produced is
consumed by AEM1-G. Four facts point the other way, and they are decisive.

**Its coupling is to the `.p` format, not to the EM problem.** It reaches into
`odas_tpw.rsi.p_file` for `PFile` and for the private `_parse_header`,
`_detect_endian` and `HEADER_BYTES` (`clocksync/extract.py:46`), and it carries
a correctness fact that only makes sense next to the reader: header word 9 is
*integer milliseconds*, unlike ODAS MATLAB's `Milli`, which `odas_p2mat.m`
writes as fractional seconds despite the name. Reading one as the other is a
factor-of-1000 error in precisely the sub-second term the package exists to
measure. Across a repo boundary that fact rots on the next `p_file.py` change,
silently.

**Nothing in the code is Seaspider-specific.** `reference.py` is deliberately
format-agnostic — MATLAB v7 and v7.3, NetCDF, or another `.p` tree — so that
the trusted clock can be *chained and verified* (CTD → Sig → MR → MR) rather
than assumed. `Signature`, `sig1000`, `MR330` and `MR429` appear only in the
config template and in docstrings, never in a code path.

**This repo already declares the consumer.** `perturb/hotel.py:643-665` takes a
`time_offset`; `perturb/config.py:75-79` documents it as an offset measured
"from pressure against pressure (`clock_offset_s`)" and pins the sign
convention. `fp07-cal fit` already measures a narrower version of the same
quantity and writes `clock_offset_s` into `coefficients.json`. Moving
`clocksync` out would leave this repo holding a partial duplicate of a
measurement whose general form had emigrated to a paper repo.

**The next MR needs it and has no EM in sight.** Every MicroRider has a clock
that jumps between `.p` files and runs at the wrong rate. RIOT `sl684`/`sl685`,
and any future MR-on-glider flown against a CTD, hit this with no EM
calibration anywhere near them.

**AEM1-G does not need it at runtime.** `make_reproduction_dataset.py:22` notes
that "offset and rate from `mr-clocksync` are applied here, so a user never
[has to re-run it]", and line 70 loads the finished
`out/clock_offsets.csv`. The published 2 Hz reproduction product already
carries the corrected clock. AEM1-G needs `clocksync` only for *raw
re-extraction* — which is exactly what its `[raw]` extra is for.

**Disposition:** land `clocksync` here (pile 1), and fix AEM1-G's `[raw]`
dependency so it can reach it (§4). One gap to close on the AEM1-G side: the
431-row `out/clock_offsets.csv` (66 kB) lives only in the SeaChest workspace.
Commit it to `AEM1-G/results/` — then AEM1-G is self-contained for the
corrected clock whether or not `clocksync` is installed, and the raw path
becomes reproducible rather than merely re-runnable.

---

## 3. Item-by-item disposition

| item | size | disposition | why |
|---|---|---|---|
| `src/odas_tpw/clocksync/`, `tests/test_clocksync.py`, `docs/clocksync/runbook.md`, `pyproject` entry point | 1712 LOC | **commit here** | §2 |
| `scripts/em_bench_zero.py` (modified, +45/−11) | tracked | **commit here** | The EM zero bench protocol is this repo's — a soak-test plan for units we fly, not an analysis of the mooring. The diff re-reduces the deployment table per leg, excludes EM 044 as not-tracking, cuts EM 046 to days 0–17, and adds the datasheet spec argument. |
| `microstructure_sensors/AEM1-G_datasheet_JFE_2018-06.pdf` | 1 file | **moved to AEM1-G** (Pat, 2026-09-06) | This row first read "commit here", on the grounds that it is the spec for a sensor we fly. Overruled: it belongs with the work that uses it. It landed under AEM1-G's own PDF policy — the file gitignored, a tracked `docs/datasheets/README.md` carrying every number — so no third-party PDF enters a repository headed for public. The `.gitignore` exception here was reverted and three pointers repointed (`em_bench_zero.py`, `papers/current-meters/README.md`, and this row). |
| `papers/current-meters/README.md` + the `papers/README.md` diff | ~20 lines | **commit here** | `papers/` tracks READMEs only — 9 tracked files, 0 PDFs — and is this repo's bibliography. EM current-meter calibration is real literature (the whole Aubrey–Trowbridge / Guza exchange), useful beyond one paper. The four PDFs stay gitignored here and are duplicated in `AEM1-G/docs/references/`; that duplication is untracked local bytes and costs nothing. |
| `docs/em_insitu_calibration_PLAN.md` | 961 lines | **DONE** — now `AEM1-G/docs/preregistration/2026-08-25_em_insitu_calibration_plan.md` | It is the plan for AEM1-G's dataset. Superseded by `AEM1-G/docs/findings/01`–`08`, but it is the only record of what was committed to *before* the results existed — the β confound, the self-wake trap, the success criteria. That is pre-registration evidence and belongs with the manuscript. AEM1-G has no copy. |
| `scripts/em_glider_scratch/` | 91 MB, 75 files | **salvage two files, then delete** | §3.1 |

### 3.1 The scratch directory

A snapshot of a live session, rescued from a volatile scratchpad. Most of it is
already superseded or regenerable:

- `glider-speed-problem.html` — **byte-identical** to the tracked
  `docs/glider_speed_problem.html`. Pure duplicate.
- The offset-analysis scripts (`em_answer.py`, `em_zero_vs_scale.py`,
  `em_zero_fit.py`, `em_offset_solve.py`, `clean_fit.py`, `origin_fit.py`,
  `excess_*.py`) are the drafts whose finished forms are
  `AEM1-G/scripts/{glider_aoa_check,slope_vs_zero,epsilon_factors}.py`.
- `sync/` (19 MB) is a `clocksync` trial run on a subset; the real run's output
  is in the SeaChest workspace. `sig_mr_probe.py` is its hand-rolled precursor,
  superseded by `mr-clocksync probe`.
- `glider-speed-problem.html` is **byte-identical** to the tracked
  `docs/glider_speed_problem.html`.

### The audit, 2026-09-06 — and a correction

An earlier revision of this section said the scratch held two files worth
salvaging and that `em_legs.py` was superseded. **Both claims were wrong, and
acting on them would have destroyed data.** Reading every script against both
trees found this dependency chain, whose two producers existed *only* here
while all four consumers are **tracked in AEM1-G**:

```
.p trees --em_extract.py--> emdata/*.npz --em_legs.py--> legs_*.npy
                                  |                          |
              glider_legs.py, glider_steadiness.py    aoa_histogram.py,
                                                      glider_aoa_check.py
```

All four consumers defaulted their `LEGS` root to the absolute path of this
scratch directory. `glider_legs.py` supersedes `em_legs.py`'s *method* — an
explicit settle window instead of "the second half" — but not its *artifact*.

**Resolved** (AEM1-G PR #4): both producers moved to
`AEM1-G/scripts/{glider_extract,glider_legs_reduce}.py`, paths routed through
`aem1g.config` (`GLIDER`, `glider_tree()`, `GLIDER_DEPS`), and the eight
products copied to `/Volumes/SeaChest/ARCTERX/Glider EM analysis/`. Verified by
regenerating all four `legs_*.npy` byte-identically. **Deleting the scratch is
now safe.**

Still open:

1. **`em_inventory.py` / `em_table.py`** — not analysis, a *tool*.
   `rsi-tpw sensors` (`rsi/sensor_inventory.py`) registers exactly two
   `SENSOR_KINDS`, shear and fp07; the **EM flowmeter is not covered**, and the
   registry comment invites new kinds. AEM1-G's `factory_cal.py` does not
   substitute: it transcribes the paper certificates, a different source from
   the `.p` config the instrument actually ran with. Worth folding in as an
   `em` kind (`serial`, `cal_date`, `a`, `b`) rather than kept as a script;
   `em_table.py`'s reusable idea is *sampling* spread-out files, since a full
   scan is ~15 min over 3000 files on a network volume.
2. **`slocum-mr-runbook.html`** ("Slocum MicroRider Runbook") — 9 sections
   walking the full chain on real data, including the wing failure and open
   questions. PR #157 landed the *configuration*
   (`examples/slocum_glider_hotel/`, `docs/dinkum_hotel.md`) but not the
   narrative, and `docs/MICRORIDER_GUIDE_PLAN.md` is still only a **plan** for
   exactly this guide. Closest thing to a draft of it.
3. **`ru33_coefficient_swap.py`**, `ru33_alpha_error.py`, `ru33_depth.py`,
   `aoa_histogram_ru33.py` — cross-repo (they import `aem1g`), so AEM1-G by the
   §1 corollary if kept at all. `ru33_coefficient_swap.py` establishes that RU33
   flew EM 042 on its *bare-sensor* certificate rather than the
   installed-on-MicroRider sheet — the opposite choice from the OSU gliders.
   Check whether `AEM1-G/src/aem1g/factory_transform.py` already captures it.

**Done:** `will_dives_work.py` landed as `scripts/will_dives_work.py` with its
forecast pre-registered in `docs/glider_dive_geometry_forecast.md`.

---

## 4. Blocking defect on the AEM1-G side

`AEM1-G/pyproject.toml` declares

```toml
raw = ["odas_tpw @ git+https://github.com/mousebrains/odas_tpw"]
```

**That repository does not exist.** The package lives in
`github.com/mousebrains/turbulence` under `src/odas_tpw/`. `pip install '.[raw]'`
fails, which disables exactly the two modules the extra exists to serve —
`extract_mr.py` and `patch_mr_time.py`, the raw-extraction path.

Fix to `git+https://github.com/mousebrains/turbulence`, and pin a tag or commit
once `clocksync` has landed here, so the paper's raw path is reproducible
against a stated version rather than against whatever `main` happens to be.

---

## 5. Order of operations

1. ~~Land `clocksync` here (package, tests, runbook, entry point, a CLAUDE.md
   subpackage entry).~~ **Done** — see the `mr-clocksync` PR. Tag the release.
2. ~~Fix AEM1-G's `[raw]` URL and pin it to that tag.~~ **Done** — AEM1-G PR #2.
   It was *two* bugs: the repository named did not exist, and the requirement
   was named for the import package (`odas_tpw`) rather than the distribution
   (`microstructure-tpw`), which pip rejects as an inconsistent name even
   against the right URL. Pinned to a commit, not a tag — turbulence has no
   tag newer than `v0.3.0`, which predates `mr-clocksync`.
3. ~~Commit `out/clock_offsets.csv` to `AEM1-G/results/`.~~ **Done** — and
   `config.clock_offsets()` prefers the tracked copy, so the reproduction path
   no longer needs the volume mounted.
4. ~~Move `docs/em_insitu_calibration_PLAN.md` to AEM1-G; delete it here.~~
   **Done** — filed as pre-registration, with a preamble naming the two places
   the results did not match the plan.
5. ~~Commit the four keepers here.~~ **Done** — PR #161, with one change of
   plan: the datasheet did *not* stay. It moved to AEM1-G under that
   repository's own PDF policy (file gitignored, a tracked
   `docs/datasheets/README.md` carrying every number), so no third-party PDF
   enters a repository headed for public. Its `.gitignore` exception here was
   reverted and three pointers repointed.
6. ~~Resolve the two salvage files, then delete `scripts/em_glider_scratch/`.~~
   **Done** — and it was far more than two files; see the audit in §3.1. All 28
   entries were dispositioned, both producers and the four RU33 scripts moved
   to AEM1-G, `em_inventory.py` became `rsi-tpw sensors --em`, and the
   products were **copied** to `/Volumes/SeaChest/ARCTERX/Glider EM analysis/`
   before the 91 MB directory was removed. AEM1-G's consumers were re-run
   afterwards to prove nothing broke.

---

## 6. What is in neither repo

The working directory
`/Volumes/SeaChest/ARCTERX/2023/Wake/Bank Seaspider EM analysis/` holds
`clocksync.yaml`, the extraction `cache/`, `out/clock_offsets.csv`, the 16 Hz
Signature extractions and the `reproduction/` product, sourced from
`.../Bank Seaspider/{MR330,MR429}`. That is correct — configs and multi-GB
intermediates are workspace, not repository — but it means **both** repos
currently depend on a mounted volume for their raw paths. Step 3 above removes
that dependency for the one artifact that is small and irreplaceable.
