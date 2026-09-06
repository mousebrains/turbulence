# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Load the reference pressure record --- the clock everything else is moved onto.

Deliberately format-agnostic.  The reference is whichever instrument's clock is
trusted, and on this mooring alone that could be the Signature (a MATLAB file),
the RBR wave gauge (NetCDF), or another MicroRider (a ``.p`` tree).  Chaining
them --- CTD to Sig, Sig to MR, MR to MR --- is how the trusted clock is
*verified* rather than assumed, so all three have to be first-class.

Time units are guessed and then stated, never assumed silently: a MATLAB
datenum (~7.4e5), a Unix epoch (~1.7e9) and days-since-1970 (~2e4) are three
orders of magnitude apart and cannot be confused, but a wrong guess would
produce a beautifully self-consistent answer about the wrong century.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from odas_tpw.clocksync.series import PressureSeries, epoch_from_datenum

# Bounds for the epoch guess, in the native units of each convention.
_DATENUM_1990, _DATENUM_2100 = 726834.0, 767011.0
_EPOCH_1990, _EPOCH_2100 = 631152000.0, 4102444800.0


def to_epoch(t: np.ndarray, units: str = "auto") -> tuple[np.ndarray, str]:
    """Return (epoch seconds, the convention that was used)."""
    t = np.asarray(t, dtype=np.float64)
    if units != "auto":
        if units == "datenum":
            return epoch_from_datenum(t), "datenum"
        if units in ("epoch", "posix", "unix"):
            return t, "epoch"
        if units == "epoch_ms":
            return t / 1000.0, "epoch_ms"
        raise ValueError(f"unknown time units {units!r}")
    med = float(np.nanmedian(t))
    if _DATENUM_1990 <= med <= _DATENUM_2100:
        return epoch_from_datenum(t), "datenum"
    if _EPOCH_1990 <= med <= _EPOCH_2100:
        return t, "epoch"
    if _EPOCH_1990 * 1000 <= med <= _EPOCH_2100 * 1000:
        return t / 1000.0, "epoch_ms"
    raise ValueError(
        f"cannot tell what time units these are (median {med:g}); "
        "set reference.time_units explicitly"
    )


def _from_matlab(path: Path, tname: str, pname: str) -> tuple[np.ndarray, np.ndarray]:
    try:  # v7.3 files are HDF5
        import h5py

        with h5py.File(path, "r") as h:
            if tname in h:
                return np.asarray(h[tname]).ravel(), np.asarray(h[pname]).ravel()
    except (OSError, ImportError):
        pass
    from scipy.io import loadmat

    d = loadmat(path, variable_names=[tname, pname])
    missing = [n for n in (tname, pname) if n not in d]
    if missing:
        raise KeyError(f"{path.name}: no variable(s) {missing}")
    return d[tname].ravel(), d[pname].ravel()


def _from_netcdf(path: Path, tname: str, pname: str) -> tuple[np.ndarray, np.ndarray]:
    import netCDF4

    with netCDF4.Dataset(path) as n:
        t = np.asarray(n[tname][:], dtype=np.float64).ravel()
        p = np.ma.filled(np.asarray(n[pname][:], dtype=np.float64), np.nan).ravel()
        units = getattr(n[tname], "units", "")
    # "<unit> since <date>" is the CF form the RBR export uses.
    if " since " in units:
        import cftime

        cal = getattr(n[tname], "calendar", "standard") if hasattr(n, "variables") else "standard"
        dates = cftime.num2date(t, units, calendar=cal, only_use_cftime_datetimes=False)
        t = np.array([d.timestamp() for d in np.atleast_1d(dates)], dtype=np.float64)
    return t, p


def load_reference(
    path: str | Path,
    *,
    kind: str = "auto",
    time_var: str = "T",
    pressure_var: str = "P",
    time_units: str = "auto",
    name: str = "reference",
    scale: float = 1.0,
    offset: float = 0.0,
) -> PressureSeries:
    """A :class:`PressureSeries` from a MATLAB, NetCDF, or ``.p`` source.

    ``scale``/``offset`` convert to dbar if the source is not already there.
    They do not move the clock and they cannot change a lag --- band-passing
    removes any constant, and a scale factor cancels out of a correlation --- so
    getting them wrong costs nothing in sync, only in reporting.
    """
    path = Path(path)
    if kind == "auto":
        sfx = path.suffix.lower()
        kind = {".mat": "matlab", ".nc": "netcdf", ".p": "pfile"}.get(sfx, "")
        if not kind:
            raise ValueError(f"cannot infer a reader for {path.name}; set reference.kind")
    if kind == "pfile":
        from odas_tpw.clocksync.extract import read_pfile_pressure

        s = read_pfile_pressure(path, name=name)
    else:
        raw_t, raw_p = (
            _from_matlab(path, time_var, pressure_var)
            if kind == "matlab"
            else _from_netcdf(path, time_var, pressure_var)
        )
        t, _ = to_epoch(raw_t, time_units)
        s = PressureSeries(t, raw_p, name, str(path))
    s.p = s.p * scale + offset
    good = np.isfinite(s.t) & np.isfinite(s.p)
    return PressureSeries(s.t[good], s.p[good], s.name, str(path), s.fs)
