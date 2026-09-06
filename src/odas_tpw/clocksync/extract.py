# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""``.p`` tree -> one small cached pressure file per source file.

Why a cache exists.  The Bank Seaspider trees are 67 GB across 431 files, and
solving a clock means visiting them repeatedly as windows, bands and gates are
tuned.  A ``PFile`` costs ~1.7 s for a 200 MB file, so a full pass is ~12
minutes --- tolerable once, wasteful every time.  Each source file reduces to a
~700 kB ``.npz`` and the whole deployment then loads in seconds.

The cache is **per source file and resumable**.  A run that dies halfway leaves
every finished file usable and picks up where it stopped, which is the rule for
anything longer than about ten minutes.  A cache entry is reused only when it
is newer than its source and carries a matching decimation rate, so changing
``rate`` in the config invalidates exactly what it should.

Decimation is an anti-aliased FIR resample, not a stride.  Plain subsampling of
a 128 Hz record to 16 Hz would fold everything from 8 to 64 Hz back into the
band, and some of that lands on top of the swell.
"""

from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from odas_tpw.clocksync.series import PressureSeries

CACHE_VERSION = 1


def pfile_start(path: str | Path) -> float:
    """Epoch seconds of a ``.p`` file's header stamp, milliseconds included.

    The 128-byte binary header carries the date; the embedded INI config does
    not.  Word 9 is integer milliseconds 0-999 --- note this is *not* the same
    as ODAS MATLAB's ``Milli``, which ``odas_p2mat.m`` writes as fractional
    seconds (``d.Milli = y(6) - d.Second``) despite the name.  Reading one as
    the other is a factor-of-1000 error in the very sub-second term this
    package exists to measure.
    """
    from odas_tpw.rsi.p_file import HEADER_BYTES, _detect_endian, _parse_header

    with open(path, "rb") as fh:
        raw = fh.read(HEADER_BYTES)
    h = _parse_header(raw, _detect_endian(raw, path))
    base = dt.datetime(
        h["year"], h["month"], h["day"], h["hour"], h["minute"], h["second"],
        tzinfo=dt.UTC,
    )
    return float(base.timestamp()) + float(h["millisecond"]) / 1000.0


def read_pfile_pressure(
    path: str | Path, rate: float = 0.0, name: str = "", channel: str = "P"
) -> PressureSeries:
    """Pressure from one ``.p`` file, optionally decimated to *rate*."""
    from odas_tpw.rsi.p_file import PFile

    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = PFile(str(path))
    if channel not in pf.channels:
        raise KeyError(f"{path.name}: no '{channel}' channel")
    p = np.asarray(pf.channels[channel], dtype=np.float64)
    fs = float(pf.fs_slow) if p.size == pf.t_slow.size else float(pf.fs_fast)
    t0 = pfile_start(path)
    if rate and rate < fs:
        q = round(fs / rate)
        if q > 1:
            # ftype='fir' with zero_phase keeps the group delay out of the
            # result; an IIR decimate would shift the record by its own phase.
            p = decimate(p, q, ftype="fir", zero_phase=True)
            fs = fs / q
    t = t0 + np.arange(p.size, dtype=np.float64) / fs
    return PressureSeries(t, p, name or path.stem, str(path), fs)


def cache_path(cache_dir: Path, src: Path) -> Path:
    return cache_dir / (src.stem + ".npz")


def extract_file(src: Path, cache_dir: Path, rate: float, channel: str = "P",
                 force: bool = False) -> Path:
    """Reduce one ``.p`` file to its cache entry; reuse a valid one."""
    out = cache_path(cache_dir, src)
    if out.exists() and not force:
        try:
            with np.load(out) as z:
                if (
                    int(z["version"]) == CACHE_VERSION
                    and float(z["fs"]) == rate
                    and float(z["mtime"]) >= src.stat().st_mtime
                ):
                    return out
        except Exception:
            pass  # unreadable or from an older layout: rebuild it
    s = read_pfile_pressure(src, rate=rate, channel=channel)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    # Write through an open handle, NOT a path: np.savez_compressed silently
    # appends ".npz" to any path that does not already end in it, so a temp
    # name like "x.npz.tmp" lands on disk as "x.npz.tmp.npz" and the rename
    # below then fails on a file that was written perfectly well.
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            version=CACHE_VERSION,
            t0=s.t[0] if s.t.size else np.nan,
            fs=s.fs,
            # float32 for pressure only: 0.001 dbar in 20 needs 15 bits, so this
            # is lossless at the instrument's resolution.  The time base is NOT
            # stored per sample -- it is regenerated from t0 and fs in float64,
            # because a float32 epoch second resolves only to ~128 s.
            p=s.p.astype(np.float32),
            mtime=src.stat().st_mtime,
            source=str(src),
        )
    tmp.replace(out)  # atomic: a killed run never leaves a half-written entry
    return out


def load_cached(path: Path, name: str = "") -> PressureSeries:
    with np.load(path, allow_pickle=False) as z:
        t0, fs, p = float(z["t0"]), float(z["fs"]), np.asarray(z["p"], dtype=np.float64)
        src = str(z["source"]) if "source" in z else ""
    t = t0 + np.arange(p.size, dtype=np.float64) / fs
    return PressureSeries(t, p, name or Path(path).stem, src, fs)


def extract_tree(
    sources: list[Path], cache_dir: Path, rate: float, channel: str = "P",
    force: bool = False, log=print,
) -> list[Path]:
    """Cache a whole tree, reporting progress; safe to interrupt and rerun."""
    cache_dir = Path(cache_dir)
    out: list[Path] = []
    n = len(sources)
    for i, src in enumerate(sorted(sources), 1):
        try:
            out.append(extract_file(src, cache_dir, rate, channel, force))
        except Exception as exc:  # one bad file must not lose the other 430
            log(f"  [{i}/{n}] {src.name}: SKIPPED -- {exc}")
            continue
        if i % 25 == 0 or i == n:
            log(f"  [{i}/{n}] cached through {src.name}")
    return out
