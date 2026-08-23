# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Read Dinkum Binary Data files into an xarray Dataset.

Three interchangeable backends, because the two that actually parse DBD are
external and neither is universally present:

``xarray-dbd``
    Pure Python (https://github.com/mousebrains/xarray-dbd). Reads
    ``*.?c?`` LZ4 files directly when the ``lz4`` package is installed.
``dbd2netcdf``
    The ``dbd2netCDF`` C++ binary (https://github.com/mousebrains/dbd2netcdf).
    Converts to a temporary NetCDF which we then open. Decompresses ``*.?c?``
    itself.
``netcdf``
    No DBD parsing at all — the inputs are already NetCDF (from a previous
    ``dbd2netCDF`` run, say). Useful for testing and for sites that convert
    once, up front.

Both DBD backends need a **sensor-list cache directory**. Slocum files
reference their sensor list by hash rather than carrying it, so a file whose
hash is not in the cache cannot be decoded at all. Left to themselves both
readers *skip* such a file and carry on, so a mission whose science files are
uncached would quietly become a flight-only hotel file. :func:`load_dinkum`
therefore compares the number of files actually decoded with the number
requested and raises, naming the skipped files, on any shortfall.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

BACKENDS = ("xarray-dbd", "dbd2netcdf", "netcdf")

# Extensions that are already NetCDF (no DBD parsing needed).
_NETCDF_SUFFIXES = frozenset({".nc", ".nc4", ".cdf"})

# dbd2netCDF writes per-file metadata on its own dimension ("j"), distinct from
# the per-record data dimension ("i"). Only the record dimension is data.
_METADATA_DIMS = frozenset({"j"})


def available_backends() -> dict[str, str]:
    """Map backend name -> why it is (un)available, for diagnostics.

    A value of ``""`` means available; anything else is the reason it is not.
    """
    out: dict[str, str] = {"netcdf": ""}

    try:
        import xarray_dbd  # noqa: F401

        out["xarray-dbd"] = ""
    except ImportError as exc:
        out["xarray-dbd"] = f"not importable ({exc})"

    if shutil.which("dbd2netCDF"):
        out["dbd2netcdf"] = ""
    else:
        out["dbd2netcdf"] = "dbd2netCDF not on PATH"
    return out


def _is_netcdf(paths: Sequence[Path]) -> bool:
    return bool(paths) and all(p.suffix.lower() in _NETCDF_SUFFIXES for p in paths)


def resolve_backend(backend: str, paths: Sequence[Path]) -> str:
    """Resolve ``"auto"`` to a concrete backend, or validate an explicit one."""
    avail = available_backends()
    if backend != "auto":
        if backend not in BACKENDS:
            raise ValueError(f"reader={backend!r}: not one of {list(BACKENDS)} or 'auto'")
        if avail[backend]:
            raise RuntimeError(f"reader={backend!r} requested but {avail[backend]}")
        return backend

    # NetCDF inputs never need a DBD parser, whatever else is installed.
    if _is_netcdf(paths):
        return "netcdf"
    for name in ("xarray-dbd", "dbd2netcdf"):
        if not avail[name]:
            return name
    raise RuntimeError(
        "No DBD reader available. Install xarray-dbd (pip install xarray-dbd) "
        "or put the dbd2netCDF binary on PATH "
        "(https://github.com/mousebrains/dbd2netcdf). Alternatively convert to "
        "NetCDF first and point files.patterns at the .nc files."
    )


def _record_dim(ds: xr.Dataset) -> str:
    """Name of the per-record dimension (the long one that isn't metadata)."""
    candidates = [d for d in ds.sizes if d not in _METADATA_DIMS]
    if not candidates:
        raise ValueError(f"No record dimension in dataset (dims: {dict(ds.sizes)})")
    # The record dimension is the largest; per-file metadata dims are tiny.
    return str(max(candidates, key=lambda d: ds.sizes[d]))


def _drop_metadata_vars(ds: xr.Dataset, dim: str) -> xr.Dataset:
    """Keep only variables that live on the record dimension."""
    keep = [str(v) for v in ds.data_vars if ds[v].dims == (dim,)]
    return ds[keep]


def _decoded_file_count(ds: xr.Dataset) -> int | None:
    """How many input files a backend actually decoded, if it says.

    xarray-dbd records ``attrs["n_files"]``; dbd2netCDF writes one row of
    ``hdr_*`` metadata per decoded file on its ``j`` dimension. ``None`` when
    neither is present (hand-made NetCDF).
    """
    n = ds.attrs.get("n_files")
    if n is not None:
        return int(n)
    if "j" in ds.sizes:
        return int(ds.sizes["j"])
    return None


def _open_netcdf(paths: Sequence[Path]) -> tuple[xr.Dataset, int | None]:
    """Open NetCDF inputs. Returns ``(dataset, n_files_decoded)``."""
    dss = []
    n_decoded: int | None = 0
    for p in paths:
        # mask_and_scale=True so that _FillValue / missing_value are honoured:
        # dbd2netCDF writes a NaN fill only for float sensors and the dtype's
        # default fill (-127, -32767, ...) for integer ones, which must not
        # survive as data. decode_times/decode_timedelta=False because Slocum
        # carries time-like (m_present_time) and duration-like
        # (m_tot_on_time) units that xarray would otherwise turn into
        # datetime64/timedelta64; we do our own epoch handling and want the
        # raw numbers.
        ds = xr.open_dataset(
            p, mask_and_scale=True, decode_times=False, decode_timedelta=False
        )
        n_here = _decoded_file_count(ds)
        n_decoded = None if (n_here is None or n_decoded is None) else n_decoded + n_here
        dim = _record_dim(ds)
        dss.append(_drop_metadata_vars(ds, dim).rename({dim: "record"}))
    if len(dss) == 1:
        return dss[0], n_decoded
    # join="outer": a sensor present in the science files but not the flight
    # files must survive the concat as NaN, not be dropped to the intersection.
    return (
        xr.concat(dss, dim="record", join="outer", combine_attrs="drop_conflicts"),
        n_decoded,
    )


_CACHE_HINT = (
    "The usual cause is a sensor-list cache miss: Slocum files reference "
    "their sensor list by hash, and a file whose hash is not in the cache "
    "directory cannot be decoded. Point files.cache at the cache directory "
    "from this glider/mission."
)


def _skipped_files(requested: Sequence[Path], n_decoded: int, detail: str = "") -> RuntimeError:
    """Build the error for *requested* files of which only *n_decoded* loaded."""
    msg = (
        f"Decoded {n_decoded} of {len(requested)} Dinkum file(s); the rest were "
        f"skipped by the reader. " + _CACHE_HINT
    )
    if detail:
        msg += f"\nReader said: {detail}"
    return RuntimeError(msg)


def _open_xarray_dbd(
    paths: Sequence[Path],
    cache: Path | None,
    to_keep: list[str] | None,
    skip_first_record: bool,
    repair: bool,
) -> xr.Dataset:
    import warnings

    import xarray_dbd as xdbd

    # xarray-dbd reports a skipped file as a UserWarning, then carries on;
    # capture them so the error can name the files it could not decode.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ds = xdbd.open_multi_dbd_dataset(
                [str(p) for p in paths],
                cache_dir=str(cache) if cache else None,
                to_keep=to_keep,
                skip_first_record=skip_first_record,
                repair=repair,
            )
        except ValueError as exc:
            # "No valid data found in any files": every file was skipped.
            raise _skipped_files(paths, 0, str(exc)) from exc
    notes = "; ".join(str(w.message) for w in caught if "rror reading" in str(w.message))
    for w in caught:
        logger.warning("xarray-dbd: %s", w.message)
    n_decoded = _decoded_file_count(ds)
    if n_decoded is not None and n_decoded < len(paths):
        raise _skipped_files(paths, n_decoded, notes)
    dim = _record_dim(ds)
    out = _drop_metadata_vars(ds, dim).rename({dim: "record"})
    out.attrs["n_files"] = n_decoded if n_decoded is not None else len(paths)
    return out


def _open_dbd2netcdf(
    paths: Sequence[Path],
    cache: Path | None,
    to_keep: list[str] | None,
    skip_first_record: bool,
    repair: bool,
) -> xr.Dataset:
    exe = shutil.which("dbd2netCDF")
    if exe is None:  # pragma: no cover - guarded by resolve_backend
        raise RuntimeError("dbd2netCDF not on PATH")

    with tempfile.TemporaryDirectory(prefix="dinkum-hotel-") as tmp:
        tmpdir = Path(tmp)
        out_nc = tmpdir / "dinkum.nc"
        # --strict: an undecodable file (cache miss, corrupt) is an error,
        # not a "skipping file" warning and a partial result.
        cmd = [exe, "--strict", "--output", str(out_nc)]
        if cache:
            cmd += ["--cache", str(cache)]
        if skip_first_record:
            # -A skips the first record of EVERY file (-s keeps the first
            # file's), matching xarray-dbd's skip_first_record.
            cmd += ["--skipAll"]
        if repair:
            cmd += ["--repair"]
        if to_keep:
            # -k takes a file of sensor names to emit; trimming here keeps the
            # temporary NetCDF small (a full Slocum sensor list is ~2000 wide).
            sensor_file = tmpdir / "sensors.txt"
            sensor_file.write_text("\n".join(to_keep) + "\n")
            cmd += ["--sensorOutput", str(sensor_file)]
        cmd += [str(p) for p in paths]

        logger.info("running %s on %d file(s)", Path(exe).name, len(paths))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            msg = f"dbd2netCDF failed (exit {proc.returncode}):\n{err or proc.stdout.strip()}"
            if "Known sensors do not include" in err or "skipping file" in err:
                msg += "\n" + _CACHE_HINT
            raise RuntimeError(msg)
        skipped = [ln.strip() for ln in err.splitlines() if "skipping file" in ln]
        for line in err.splitlines():
            if "warning" in line.lower():
                logger.warning("dbd2netCDF: %s", line.strip())
        if not out_nc.exists():
            raise RuntimeError("dbd2netCDF produced no output file")
        # Materialize before the temp dir evaporates.
        ds, n_decoded = _open_netcdf([out_nc])
        ds = ds.load()
        if n_decoded is None:
            n_decoded = len(paths) - len(skipped)
        if skipped or n_decoded < len(paths):
            raise _skipped_files(paths, n_decoded, "; ".join(skipped))
        ds.attrs["n_files"] = n_decoded
        return ds


def load_dinkum(
    paths: Iterable[str | Path],
    *,
    backend: str = "auto",
    cache: str | Path | None = None,
    sensors: Sequence[str] | None = None,
    skip_first_record: bool = True,
    repair: bool = False,
) -> xr.Dataset:
    """Read Dinkum files into one Dataset on a ``record`` dimension.

    Parameters
    ----------
    paths : iterable of path
        DBD/EBD files (compressed or not), or NetCDF files.
    backend : str
        ``"auto"`` (default), ``"xarray-dbd"``, ``"dbd2netcdf"``, or
        ``"netcdf"``. See :func:`resolve_backend`.
    cache : path, optional
        Sensor-list cache directory. Required in practice for DBD input —
        a file whose sensor-list hash is not cached cannot be decoded.
    sensors : sequence of str, optional
        Restrict the read to these sensor names (plus whatever the caller
        needs). ``None`` reads every sensor. Pass the union of the data
        sensors *and* every time sensor: dropping a time sensor here makes
        the projection impossible later.
    skip_first_record : bool
        Skip each file's first record (the first record of a file is
        routinely partial). Passed as ``skip_first_record`` to xarray-dbd and
        as ``-A``/``--skipAll`` to dbd2netCDF; neither reader skips it on
        its own.
    repair : bool
        Attempt recovery of corrupted records rather than skipping them.

    Returns
    -------
    xr.Dataset
        All requested sensors on a single ``record`` dimension, float64,
        absent values as NaN.
    """
    file_paths: list[Path] = [Path(p) for p in paths]
    if not file_paths:
        raise ValueError("No input files")
    missing = [p for p in file_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {[str(p) for p in missing[:5]]}")

    resolved = resolve_backend(backend, file_paths)
    to_keep = sorted(set(sensors)) if sensors else None
    logger.info("reading %d Dinkum file(s) with backend %r", len(file_paths), resolved)

    n_decoded: int | None
    if resolved == "netcdf":
        ds, n_decoded = _open_netcdf(file_paths)
        if n_decoded is None:
            n_decoded = len(file_paths)
    elif resolved == "xarray-dbd":
        ds = _open_xarray_dbd(file_paths, _as_path(cache), to_keep, skip_first_record, repair)
        n_decoded = int(ds.attrs["n_files"])
    else:
        ds = _open_dbd2netcdf(file_paths, _as_path(cache), to_keep, skip_first_record, repair)
        n_decoded = int(ds.attrs["n_files"])

    n = ds.sizes.get("record", 0)
    if n == 0:
        raise RuntimeError(
            f"Backend {resolved!r} decoded 0 records from {len(file_paths)} file(s). "
            + _CACHE_HINT
        )
    ds.attrs.pop("n_files", None)
    ds.attrs["dinkum_reader"] = resolved
    ds.attrs["dinkum_source_files"] = int(n_decoded)
    ds.attrs["dinkum_requested_files"] = len(file_paths)
    logger.info("read %d records, %d sensors", n, len(ds.data_vars))
    return ds


def _as_path(v: str | Path | None) -> Path | None:
    return Path(v) if v is not None else None


def sensor_inventory(ds: xr.Dataset) -> list[dict]:
    """Per-sensor summary of a loaded Dataset, for authoring the YAML.

    Returns one dict per sensor with ``name``, ``units``, ``n_finite``,
    ``fraction``, ``min``, ``max`` — sorted by name. Sensors that never
    reported (all-NaN) are included with ``n_finite == 0``, since their
    absence is exactly what you want to see before listing them.
    """
    out: list[dict] = []
    n_rec = ds.sizes.get("record", 0)
    for name in sorted(str(v) for v in ds.data_vars):
        arr = np.asarray(ds[name].values, dtype=np.float64)
        finite = np.isfinite(arr)
        n_finite = int(finite.sum())
        out.append(
            {
                "name": name,
                "units": str(ds[name].attrs.get("units", "")),
                "n_finite": n_finite,
                "fraction": (n_finite / n_rec) if n_rec else 0.0,
                "min": float(np.min(arr[finite])) if n_finite else float("nan"),
                "max": float(np.max(arr[finite])) if n_finite else float("nan"),
            }
        )
    return out
