#!/usr/bin/env python3
"""Re-frame MicroRider SN 156's 2022 .P files: drop the damaged lead-in record(s).

    python reframe_p.py          # ../post-recovery/microrider/*.P -> reframed/microrider/*.P
                                 #   + reframed/reframe_manifest.csv
    python reframe_p.py --check  # rebuild in memory and compare with the manifest's md5s

WHY.  A v6 .P file is record 0 (header + config) followed by fixed-size data
records: a 128-byte header then 64 address-matrix cycles ("matrices", 64 words
each, every one starting with the special character 0x7FF0 of channel 255).
Readers (PFile, ODAS read_odas) slice the file at that fixed size. 28 of the 29
files of this deployment contain ONE damaged spot, in data records #4-#17 (#74
in the deck file 211_025): a record holding 33, 67, 70, 93 or 98 matrices
instead of 64, sometimes with the next record's header gone and/or a few junk
words. 0.47-0.95 s of samples are missing there (the byte counts say exactly
how much). Everything after it is framed correctly (64 matrices, consecutive
record numbers, header word 16 = 0). Read at fixed size, though, every header
after the damage lands inside the data and vice versa: PFile warns (bad-buffer
flags in ~90% of records, record-number "gaps", non-integer record count) and
returns mis-assembled channels (211_029: P from -42.8 to -22.1 dbar).

WHAT.  Walk each file by header signature (file number, header version, header
and record size, matrix shape) and matrix runs, find the last irregular record,
and write a copy holding record 0 + the data records AFTER it, byte-for-byte.
Nothing is interpolated or padded. The dropped lead-in is 5-17 s at the surface
after "acquisition start" (75 s in 211_025, a deck record). An irregular FINAL
record alone is left in place: PFile drops a trailing partial record itself.

TIME.  PFile/ODAS put data record #1, sample 0, at the record-0 header time
minus recsize, and every later sample at its index / fs_fast. The header times
continue at one per second across the damage (e.g. 211_040: #13 at 12:04:03,
#15 at 12:04:05), so sampling ran on and only the write lost bytes. The copy's
record-0 time is therefore advanced by (k - 1) true record durations
(data_words / f_clock), k = the first kept record number, which leaves every
kept sample at its original nominal time (rounded to the header's 1 ms). The
helper is PFile's own (the one cutp uses). The MR clock is still solved per
file against glider pressure (mr_clock.py), as for every MicroRider here.

The originals are not touched. Files with no damage (211_034) are copied whole.
"""
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from odas_tpw.rsi.p_file import _advance_record0_timestamp

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "post-recovery" / "microrider"
OUT = HERE / "reframed" / "microrider"
MANIFEST = HERE / "reframed" / "reframe_manifest.csv"
SPECIAL = 0x7FF0
HDR_WORDS = 64
MATRIX_WORDS = 64        # 8 rows x 8 columns
MATRICES_PER_RECORD = 64  # 512 scans per 1-s record
SIGNATURE = (0, 10, 17, 18, 28, 29, 30)  # file number, version, sizes, matrix shape


def walk(words, h0):
    """Records as (start_word, record_number, n_matrices, n_junk_words).

    *words* is the file after record 0, as big-endian uint16 aligned to the
    first data record (record 0 can hold an odd number of config bytes)."""
    def is_header(q):
        return q + HDR_WORDS <= words.size and all(words[q + i] == h0[i] for i in SIGNATURE)

    recs, pos = [], 0
    while pos < words.size:
        if not is_header(pos):
            raise SystemExit(f"expected a record header at data word {pos}")
        q, m, junk = pos + HDR_WORDS, 0, 0
        while q < words.size and not is_header(q):
            nxt = q + MATRIX_WORDS
            if words[q] == SPECIAL and nxt <= words.size and (nxt == words.size or words[nxt] == SPECIAL or is_header(nxt)):
                m, q = m + 1, nxt
                continue
            q, junk = q + 1, junk + 1
        recs.append((pos, int(words[pos + 1]), m, junk))
        pos = q
    return recs


def reframe(src):
    """(output bytes, manifest row) for one source file."""
    raw = src.read_bytes()
    h0 = np.frombuffer(raw[: 2 * HDR_WORDS], dtype=">u2")
    if int(h0[17]) != 2 * HDR_WORDS or int(h0[18]) != 2 * (HDR_WORDS + MATRICES_PER_RECORD * MATRIX_WORDS):
        raise SystemExit(f"{src.name}: unexpected header/record size {h0[17]}/{h0[18]}")
    first = int(h0[17]) + int(h0[11])
    words = np.frombuffer(raw, dtype=">u2", offset=first, count=(len(raw) - first) // 2)
    recs = walk(words, h0)
    if not recs:
        raise SystemExit(f"{src.name}: no data records")
    # Irregular: not 64 matrices, junk words, or the NEXT record number is not
    # this one + 1 (the following record's header was lost into it). A final
    # irregular record is a trailing partial record, which PFile drops itself.
    bad = [i for i, (_, num, m, junk) in enumerate(recs[:-1])
           if m != MATRICES_PER_RECORD or junk or recs[i + 1][1] != num + 1]
    start = datetime(*(int(x) for x in h0[3:9]), int(h0[9]) * 1000)
    row = dict(file=src.name, n_records_in=len(recs), irregular=";".join(
        f"#{recs[i][1]}:{recs[i][2]}m+{recs[i][3]}j" for i in bad), first_kept=f"#{recs[0][1]}",
        dropped_records=0, dropped_s="", start_in=start.isoformat(timespec="milliseconds"),
        start_out=start.isoformat(timespec="milliseconds"), md5_in=hashlib.md5(raw).hexdigest(), md5_out="")
    if not bad:
        row["md5_out"] = row["md5_in"]
        return raw, row
    if recs[0][1] != 1:
        raise SystemExit(f"{src.name}: first data record is #{recs[0][1]}, not #1; cannot place the kept records in time")
    keep = bad[-1] + 1
    tail = recs[keep:]
    if any(m != MATRICES_PER_RECORD or junk for _, _, m, junk in tail[:-1]):
        raise SystemExit(f"{src.name}: irregular record after the last one flagged")
    k = tail[0][1]
    if any(num != k + i for i, (_, num, _, _) in enumerate(tail)):
        raise SystemExit(f"{src.name}: kept record numbers are not consecutive")
    # Data record #k is 0-based index k - 1: advance by k - 1 TRUE record durations.
    rec0 = _advance_record0_timestamp(raw[:first], ">", k - 1)
    out = rec0 + raw[first + 2 * tail[0][0]:]
    h1 = np.frombuffer(rec0[: 2 * HDR_WORDS], dtype=">u2")
    new = datetime(*(int(x) for x in h1[3:9]), int(h1[9]) * 1000)
    row.update(first_kept=f"#{k}", dropped_records=keep, dropped_s=f"{(new - start).total_seconds():.3f}",
               start_out=new.isoformat(timespec="milliseconds"), md5_out=hashlib.md5(out).hexdigest())
    return out, row


def main():
    sources = sorted(SRC.glob("*.P"))
    if not sources:
        raise SystemExit(f"no .P files in {SRC}")
    if "--check" in sys.argv[1:]:
        expected = {r["file"]: r["md5_out"] for r in csv.DictReader(open(MANIFEST))}
        diffs = [s.name for s in sources if reframe(s)[1]["md5_out"] != expected.get(s.name)]
        print(f"{len(sources) - len(diffs)} of {len(sources)} match {MANIFEST.name}; differ: {diffs}")
        return 1 if diffs else 0
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in sources:
        out, row = reframe(src)
        tmp = (OUT / src.name).with_suffix(".tmp")
        tmp.write_bytes(out)
        tmp.replace(OUT / src.name)
        rows.append(row)
        print(f"{src.name}: {row['irregular'] or 'clean'} -> keep from {row['first_kept']} "
              f"({row['dropped_s'] or 0} s dropped), start {row['start_in']} -> {row['start_out']}", flush=True)
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {MANIFEST.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
