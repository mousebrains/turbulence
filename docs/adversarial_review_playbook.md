# Adversarial review playbook

How to run a deep, publication-grade review of this repository with a fan-out of
reviewer agents. Recovered 2026-09-06 from the harness that produced
[issue #104](https://github.com/mousebrains/turbulence/issues/104), which until
then existed only in a scratch directory outside version control.

This is the **heavyweight** process: days of agent time, aimed at "could a
referee falsify a number in our paper?" For an ordinary change, `/code-review`
is the right tool and this is overkill. Reach for this before a submission,
before a release, or when a subsystem has changed enough that you no longer
know what is true about it.

---

## 1. The reviewer contract

Every reviewer gets these instructions. They are the part that matters most —
the unit briefs are just scope.

- **You are an adversarial reviewer for publication-grade software.** Attack the
  code: construct counterexamples and *run* them; verify math against the cited
  literature (`papers/` holds the PDFs); check claimed constants and equations
  **to the digit**. Prefer depth over breadth within your unit.
- **Severity scale.** CRITICAL (wrong published numbers), HIGH (wrong results in
  plausible configurations), MEDIUM (misleading docs/metadata, latent traps),
  LOW (hygiene).
- **Every finding carries:** severity, a one-line defect statement, `file:line`,
  a reproduction (code *and* observed output), and a suggested fix. A finding
  without a reproduction is a hypothesis, and it should say so.
- **Write findings incrementally to disk.** Create `findings/<UNIT>.md` as your
  *first* action, with a header and a checklist of what you plan to attack, then
  append each finding the moment you confirm it. Sessions get killed by usage
  limits at arbitrary moments; anything not on disk is lost.
- **End with an "attacked-but-sound" list**, then the literal line
  `UNIT COMPLETE`. What held is evidence too, and without it you cannot tell a
  clean subsystem from an unreviewed one.
- **Return to the caller a ≤10-line summary only.** The findings file is the
  deliverable; the conversation is not.
- **Read-only on the repository and on `/Volumes/SeaChest`.** Never
  `git add`/`commit`/`push`.

## 2. Partition into units

One reviewer per subsystem, sized so a reviewer can go deep rather than skim.
The July 2026 partition, which still maps onto the tree:

| unit | scope |
|---|---|
| U1 rsi-io | `.p` parsing and TN-051 conformance, `channels.py` conversions vs the `odas/` MATLAB reference, NetCDF fidelity, profile-detection edge cases |
| U2 epsilon-core | `spectral.py` (Parseval by construction), `goodman.py`, `despike.py`, `nasmyth.py` vs Lueck 2022, L3/L4 integration limits and FM/fom, `ocean.py` |
| U3 chi-core | Methods 1/2, Batchelor/Kraichnan forms vs the cited PDFs, `k_B` and Pr factors, the `6·κ·I` integral, `fp07.py` transfer function and noise model |
| U4 processing | `mixing.py` TEOS-10 usage and masking, `thorpe.py` vs Kaminski 2021, the combiners' Lueck 2022 Part I eq 11, `ct_align`, `top_trim`, `bottom` |
| U5 perturb-pipeline | end-to-end flow, QC ordering, probe exclusion, atomic writes, parallel workers, config canonicalization, binning conventions, CF compliance |
| U6 perturb-plot | *could any plot misrepresent the data* — axis math, great-circle distance, silent drops, mislabeled units |
| U7 docs-accuracy | every equation, constant, default and quoted result in `docs/` and `CLAUDE.md`, recomputed; every DOI resolved; every citation's volume/pages |
| U8 cross-cutting | unit and sign conventions end-to-end (dbar vs m, positive-down, W/kg, cpm vs rad/m), the ATOMIX benchmark, test blind spots, determinism under parallelism |

Two scoping rules that earned their place:

- **Name what is excluded, and why.** `pyturb/` is Jesse's hosted code — note
  issues, do not review it (`CLAUDE.md`, *Subpackages*).
- **U7 is not optional.** A referee reads the docs. Quoted numbers that no
  longer match the code are the cheapest way to lose credibility, and they are
  invisible to every test.

## 3. Verify independently — and read the confirm rate

`V1`: for **every** finding, independently reproduce it. Mark CONFIRMED with the
reproduction, or REFUTED with why.

> Publication stakes: a false accusation wastes the author's time; a missed real
> defect corrupts a paper. Be ruthless in both directions.

**What actually happened in July 2026, and the caveat it carries.** Counting
verdict words across the three verifier files gives **109 "confirmed", 8
"partial", 1 "refuted", 1 "rejected"** — a word count, not a finding count, since
the unit files use inconsistent finding IDs and some verdicts are restated in
prose. The ratio is the point, and it is lopsided. A verify pass that overturns
roughly one finding in a hundred is not
obviously working, and there are two readings:

- the reviewers were disciplined — they were told to report only what they had
  *run*, so little survived to the verifier that was wrong; or
- the verifier was insufficiently independent, and confirmed what it was handed.

The evidence cannot separate these from here. What the verifiers demonstrably
*did* do is **re-scope severity** — "the finding's 14–29 % headline overstates
the typical published impact", "CONFIRMED (static). LOW cosmetic (title only)"
— which may be the pass's real function. If you rerun this: give the verifier a
brief that asks explicitly for severity re-scoring as a first-class verdict, and
seed it with a couple of deliberately wrong findings to check it can still say
no.

## 4. Iterate where the defects are

`I2`: take the three units with the most confirmed findings and spawn **fresh**
reviewers with narrower, deeper briefs informed by what was found — defects
cluster, and the second pass into a bad neighborhood is the highest-yield one.
Sweep anything no unit covered (scripts, CI config, examples) in the same round.

## 5. Assemble from confirmed findings only

`A1`: build the issue body from the verified set — never from the raw unit
files. Sections: Scope & methodology; Critical/High (each with defect, evidence
`file:line`, reproduction, suggested fix); Medium; Low/hygiene as a compact
list; Documentation accuracy; **What was attacked and held**; Recommendations.

File it with `gh issue create`, and record the URL in the run's `STATE.md`.

## 6. Operational rules

- **A `STATE.md` per run**, updated as units complete. It is what a resumed
  session reads to find out what is already done.
- **Survive the quota window.** A full run outlives a single session. Drive it
  from a self-terminating cron chain that *resumes* rather than restarts, and
  make every step idempotent — see the quota-resume note in the working memory.
- **Findings live on disk, in their own directory, from the first minute.**
- **The run directory is disposable; its outputs are not.** Everything durable
  belongs in the issue, or in `docs/`. The July run's directory sat outside git
  for two months holding the only copy of this playbook, which is why it is
  here now.

## 7. What the July 2026 run produced

Ten unit files (U1–U8, I2a, I2b), verified in three passes (V1a/b/c) and
assembled into issue #104 — closed. The follow-on
MicroRider-readiness review became
[#131](https://github.com/mousebrains/turbulence/issues/131), resolved by
PRs #132–#140 and #142–#143, and still open for five deferred items.

The single most useful structural habit, worth keeping whatever else changes:
**"attacked-but-sound" is recorded alongside the defects.** It is what lets a
later reader tell "this was checked and is fine" from "nobody looked."
