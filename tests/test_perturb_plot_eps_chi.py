# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the eps-chi plot preset's per-profile attr lookup.

Focus: the title's processing-parameter metadata (fft/diss length, spectrum
model, ...) must come from the *config-resolved* per-profile ``diss`` dir, not
from a newest-``diss_NN`` glob. In a multi-config output_root the latter would
caption old combo data with a newer run's parameters.
"""

import xarray as xr

from odas_tpw.perturb import config as cfg
from odas_tpw.perturb import resolve
from odas_tpw.perturb.plot import eps_chi


def _write_diss(root, config, seq, attrs):
    """Create ``diss_{seq:02d}`` with a signed per-profile NetCDF carrying *attrs*."""
    d = root / f"diss_{seq:02d}"
    d.mkdir()
    section, params, upstream = cfg.stage_signature("diss", config)
    cfg.write_signature(d, section, params, upstream=upstream)
    ds = xr.Dataset(attrs=attrs)
    ds.to_netcdf(d / "x_prof001.nc")
    return d


class TestPerProfileAttrs:
    def test_none_directory(self):
        assert eps_chi._per_profile_attrs(None) == {}

    def test_empty_directory(self, tmp_path):
        assert eps_chi._per_profile_attrs(str(tmp_path)) == {}

    def test_reads_attrs(self, tmp_path):
        ds = xr.Dataset(attrs={"fft_length": 256, "fs_fast": 512.0})
        ds.to_netcdf(tmp_path / "a_prof001.nc")
        got = eps_chi._per_profile_attrs(str(tmp_path))
        assert got["fft_length"] == 256
        assert got["fs_fast"] == 512.0

    def test_config_resolved_dir_not_newest(self, tmp_path):
        """The regression: two diss runs under one root with different fft_length.
        Resolving the OLD config must read the OLD dir's attrs (256), never the
        newest dir's (512)."""
        c_old = {"files": {"output_root": str(tmp_path)},
                 "epsilon": {"fft_length": 256}}
        c_new = {"files": {"output_root": str(tmp_path)},
                 "epsilon": {"fft_length": 512}}
        _write_diss(tmp_path, c_old, 0, {"fft_length": 256, "fs_fast": 512.0})
        _write_diss(tmp_path, c_new, 1, {"fft_length": 512, "fs_fast": 512.0})

        resolved = resolve.stage_dir(c_old, "diss")
        assert resolved.name == "diss_00"  # config picks the old dir, not newest
        assert eps_chi._per_profile_attrs(str(resolved))["fft_length"] == 256
