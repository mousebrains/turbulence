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
