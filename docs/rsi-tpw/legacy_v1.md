# Legacy header-v1 `.p` files (`rsi-tpw v1to6`)

Pre-2015 Rockland instruments (e.g. the VMP-2000, SN002, Taiwan 2013 corpus of
GitHub issue #141) write **ODAS header-version-1** `.p` files. They differ
from modern (v6+) files in exactly one place — record 0:

| | v6+ (modern) | v1 (legacy) |
|---|---|---|
| header word 11 | `0x06NN` (major 6, minor NN) | `1` |
| record 0 | 128-byte header + embedded INI configuration | full-size record: 128-byte header + **binary address matrix** (row-major uint16, zero-padded) |
| first record size | `header_size + config_size` | `record_size` |
| configuration | embedded in the file | **external setup file** (old `key: values` dialect) |
| data records | identical: 128-byte header + int16 samples in matrix scan order | identical |

Reading a v1 file with v6 record arithmetic mis-tiles the whole file (the
"garbage header words / 5632-byte tail" symptoms of issue #141). Every
`rsi-tpw`/perturb tool therefore either translates v1 files or refuses them
loudly with a remedy.

## The translator: translate once, then use every existing tool

```bash
rsi-tpw v1to6 VMP_002/TAI_013_*.p -o translated/ --sens sh1=0.0893,sh2=0.0558
rsi-tpw info translated/TAI_013_002.p        # a normal v6 file from here on
rsi-tpw eps  translated/TAI_013_002.p -o epsilon/
```

`v1to6` synthesizes a v6 INI configuration from the setup file, writes a new
record 0 (vendor `patch_setupstr.m` header contract: word 11 → `0x0600`, word
12 → config size, words 13/14 → 0, **all other header words unchanged**), and
copies every data record **byte-for-byte** — lossless, timestamps and
bad-buffer flags included. (Deliberate deviation from the vendor tool, which
overwrites record 1's header and the original file.) Provenance is embedded
as machine-readable `[root]` keys in the synthesized INI: `translated_from`,
`v1_source_file`, `setup_file_source`, `setup_file_md5`, `sens_source`,
`translator`, `translated_on`.

Recommended workflow for a v1 corpus: translate once, archive the translated
v6 copies next to (never over) the originals, and process those.

`PFile` also reads raw v1 files **directly** — it performs the same
translation in memory and parses the result with the unchanged v6 reader
(`pf.translated_from_v1`, `pf.setup_file_source` record the provenance). Use
`PFile(path, setup_file=...)` to override the auto-detected setup file.

## Setup-file discovery and dialects

A v1 file has no embedded configuration, so one is found for it:

1. an explicit `--setup-file PATH` / `setup_file=` always wins;
2. otherwise siblings of the `.p` file are searched, then one level up, in
   this order (case-insensitive): `setup.txt`, `setup*.txt`, `setup*.cfg`;
3. nothing found → a loud error naming what was searched.

`setup.txt` ranks first deliberately: for the 2013 Taiwan corpus, the
acquisition log proves `setup.txt` (= `SetUp2013.txt`) drove acquisition,
while the sibling library-3.1-era `.cfg` files carry a **wrong** pressure
polynomial and matrix. Candidates in the INI dialect (`[section]` headers)
are sniffed and parsed as such; all parseable candidates are cross-checked
against the chosen one (matrix + pressure polynomial) with warnings on
disagreement, and the chosen file's matrix must equal the **binary matrix in
record 0** — a mismatch aborts the translation.

The old dialect has no channel types; sensor identity comes from the fixed
old-ODAS address map (0=Gnd, 1-3=Ax/Ay/Az, 4-7=FP07 pairs, 8/9=sh1/sh2,
10/11=P/P_dP, 12=C1_dC1, 16/17 and 18/19 = Sea-Bird even/odd pairs,
255=sp_char), verified channel-by-channel against the 2013 ground-truth
products (issue #141).

## Shear-probe sensitivities (`sens`)

**No v1 setup file carries shear sensitivities** — in 2013 they lived on
paper calibration sheets, and probes were swapped mid-cruise. A shear channel
with no `sens` is a **hard error at conversion time** (never a silent
default: a wrong sens scales epsilon by sens⁻²). Three ways to supply it:

1. `rsi-tpw v1to6 --sens sh1=0.0893,sh2=0.0558` at translation time;
2. `sh1_sens:`/`sh2_sens:` keys (generic `<name>_sens:`, a documented dialect
   extension) added to a per-epoch copy of the setup file;
3. `rsi-tpw patch-config --add-keys` on the **translated** files — the
   primary workflow for per-epoch corrections, since translated files are
   ordinary v6 files and the whole `patch-template`/`patch-config` tooling
   applies. `rsi-tpw sensors --cal-dir` then cross-checks the values.

Process each sens epoch separately (translate each cast range with its own
`--sens` values); the per-epoch values for the Taiwan 2013 corpus are
recorded on issue #141.

## Temperature on v1 corpora

v1 FP07 thermistors are kept as **raw counts**: the acquisition setup files
carry no thermistor coefficients, and the only on-disk candidates reproduce
the 2013 product's response shape but sit ~4.2 °C off its in-situ-calibrated
temperature — not trustworthy enough to publish as physical units. The
Sea-Bird SBE3 (`sbt` type, converted with vendor-parity coefficients,
verified to ≤1e-8 °C) is the **right reference temperature** on these
corpora: the `auto` reference-temperature chain tries `sbt`-type channels
(by type, after `JAC_T`) once the FP07 counts fail plausibility QC, and
`--temperature SBT1` selects it explicitly. Measured salinity works via
`--conductivity SBC1 --salinity measured` (the `sbc` converter outputs
mS/cm, directly compatible with TEOS-10 `SP_from_C`). Chi on v1 corpora is
deferred until a trusted thermistor calibration exists (in-situ FP07
calibration against the SBE3 is the natural follow-up).

## Other v1 notes

- FP07 base channels (addr 4/6) occupy two slow slots per cycle (128 Hz);
  extraction keeps the first slot per cycle at the slow rate with a warning
  (the existing k-occurrence path). Full-rate interleaved extraction is a
  possible follow-up.
- `P_dP` keeps its own polynomial and no `diff_gain` (the pressure
  pre-emphasis gain is not in the acquisition setup file), so pressure comes
  from the plain `P` channel — verified to ≤1e-9 dBar against 2013 products.
- Bad-buffer records (header word 16) warn on read exactly as for v6 files;
  restart flags (word 17) are unwarned, matching the v6 path.
- Big-endian v1 files are handled per spec but **unverified** (no such
  corpus exists here); the translator warns on one.
- Instrument identity: v1 setup files record none. Add `model:`/`sn:`/
  `vehicle:` extension keys to the setup file if known; `vehicle` defaults
  to `vmp`.

## Known limitation: reading a raw v1 file whose setup lacks `sens`

`info`/`nc`/`prof`/`eps`/`chi` do not take `--setup-file` — direct reads of raw
v1 files rely on sibling auto-detection (the Python API accepts
`PFile(..., setup_file=...)`). If the only discoverable setup file lacks probe
sensitivities and the media is read-only, translate first:
`rsi-tpw v1to6 FILE --setup-file /path/to/edited_setup.txt -o out/` (or
`--sens sh1=...,sh2=...`) and process the translated copy — the intended
workflow. Per-command `--setup-file` flags are a possible follow-up (issue #141).
