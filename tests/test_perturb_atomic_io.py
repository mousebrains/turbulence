# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.atomic_io — crash-atomic NetCDF writes (#104 U5-2)."""

import os

import numpy as np
import pytest
import xarray as xr

from odas_tpw.perturb.atomic_io import atomic_to_netcdf, tmp_sibling


class TestTmpSibling:
    def test_is_hidden_pid_suffixed_sibling(self, tmp_path):
        out = tmp_path / "a.nc"
        tmp = tmp_sibling(out)
        assert tmp.parent == out.parent  # same directory (hence same filesystem)
        assert tmp.name == f".{out.name}.{os.getpid()}.tmp"


class TestAtomicToNetcdf:
    def test_success_writes_and_leaves_no_tmp(self, tmp_path):
        ds = xr.Dataset({"x": (("t",), np.arange(5.0))})
        out = tmp_path / "a.nc"
        atomic_to_netcdf(ds, out)
        assert out.exists()
        assert not list(tmp_path.glob(".a.nc.*.tmp"))

    def test_forwards_kwargs(self, tmp_path):
        """encoding= (and any to_netcdf kwarg) must reach xarray — CTD relies on
        it to strip coord _FillValue for CF compliance."""
        ds = xr.Dataset(coords={"t": np.arange(3.0)}, data_vars={"x": (("t",), np.ones(3))})
        out = tmp_path / "enc.nc"
        atomic_to_netcdf(ds, out, encoding={"t": {"_FillValue": None}})
        with xr.open_dataset(out) as got:
            assert got["t"].encoding.get("_FillValue") is None

    def test_failure_leaves_no_partial_and_no_tmp(self, tmp_path, monkeypatch):
        ds = xr.Dataset({"x": (("t",), np.arange(5.0))})
        out = tmp_path / "b.nc"

        def boom(self, *a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(xr.Dataset, "to_netcdf", boom)
        with pytest.raises(OSError):
            atomic_to_netcdf(ds, out)
        assert not out.exists()  # no partial at the live path
        assert not list(tmp_path.glob(".b.nc.*.tmp"))  # temp cleaned


# --- bounded retry for transient network-filesystem faults -------------------


def _hdf_error(path="x.nc"):
    """The exact shape netCDF-C raises for the SeaChest SMB transient."""
    return OSError(-101, "NetCDF: HDF error", path)


def test_retry_transient_io_recovers_after_transient_failures():
    from odas_tpw.perturb.atomic_io import retry_transient_io

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _hdf_error()
        return "ok"

    assert retry_transient_io(flaky, what="test") == "ok"
    assert calls["n"] == 3


def test_retry_transient_io_gives_up_after_a_bounded_number_of_attempts():
    import pytest

    from odas_tpw.perturb.atomic_io import _TRANSIENT_ATTEMPTS, retry_transient_io

    calls = {"n": 0}

    def always_bad():
        calls["n"] += 1
        raise _hdf_error()

    with pytest.raises(OSError):
        retry_transient_io(always_bad, what="test")
    assert calls["n"] == _TRANSIENT_ATTEMPTS


def test_retry_transient_io_does_not_retry_a_real_error():
    """A missing file is an OSError subclass but is NOT transient. Retrying it
    would hide a real bug behind a multi-second delay."""
    import pytest

    from odas_tpw.perturb.atomic_io import retry_transient_io

    calls = {"n": 0}

    def missing():
        calls["n"] += 1
        raise FileNotFoundError(2, "No such file or directory", "nope.nc")

    with pytest.raises(FileNotFoundError):
        retry_transient_io(missing, what="test")
    assert calls["n"] == 1, "a genuine error must not be retried"


def test_is_transient_io_classification():
    from odas_tpw.perturb.atomic_io import _is_transient_io

    assert _is_transient_io(_hdf_error())
    assert _is_transient_io(OSError(-51, "NetCDF: Unknown file format", "x.nc"))
    assert not _is_transient_io(FileNotFoundError(2, "missing"))
    assert not _is_transient_io(PermissionError(13, "denied"))
    assert not _is_transient_io(ValueError("not even an OSError"))


def test_retry_transient_io_warns_on_each_retry(caplog):
    """Silent retries would hide a degrading mount, which is the one thing the
    retry count is useful for."""
    import logging

    from odas_tpw.perturb.atomic_io import retry_transient_io

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _hdf_error()
        return 1

    with caplog.at_level(logging.WARNING):
        retry_transient_io(flaky, what="some/path.nc")
    assert any("transient I/O" in r.getMessage() for r in caplog.records)
