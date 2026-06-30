# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb's code-aware incremental processing cache.

Covers the engine fingerprint (Part A), per-file skip markers (Part B), the
bin/combo manifest skip (Part C), and ``--force`` (Part D).
"""

import json
import os
from pathlib import Path

import pytest

from odas_tpw.perturb import config as cfg
from odas_tpw.perturb import pipeline as pl
from odas_tpw.perturb import resolve


@pytest.fixture(autouse=True)
def _reset_engine_fingerprint_cache():
    """Keep the lru_cached fingerprint from leaking a pinned override between
    tests (some tests set $ODAS_TPW_ENGINE_FINGERPRINT)."""
    cfg.engine_fingerprint.cache_clear()
    yield
    cfg.engine_fingerprint.cache_clear()


# ---------------------------------------------------------------------------
# Part A — engine fingerprint
# ---------------------------------------------------------------------------

class TestEngineFingerprint:
    def test_deterministic(self):
        cfg.engine_fingerprint.cache_clear()
        assert cfg.engine_fingerprint() == cfg.engine_fingerprint()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ODAS_TPW_ENGINE_FINGERPRINT", "pinned-value")
        cfg.engine_fingerprint.cache_clear()
        assert cfg.engine_fingerprint() == "pinned-value"
        cfg.engine_fingerprint.cache_clear()  # don't leak into other tests

    def test_injected_into_signature(self, monkeypatch):
        monkeypatch.setenv("ODAS_TPW_ENGINE_FINGERPRINT", "ENGINE-X")
        cfg.engine_fingerprint.cache_clear()
        canon = json.loads(cfg.canonicalize("epsilon", {"fft_length": 512}, upstream=[]))
        assert canon["_engine"] == {"fingerprint": "ENGINE-X"}
        cfg.engine_fingerprint.cache_clear()

    def test_code_change_yields_new_dir_but_plot_still_matches(self, monkeypatch):
        """A different engine => different dir-selection hash (recompute), yet the
        plot resolver still matches on config (it strips _engine)."""
        base = {"files": {"output_root": "/x"}}

        def _sig(engine):
            monkeypatch.setenv("ODAS_TPW_ENGINE_FINGERPRINT", engine)
            cfg.engine_fingerprint.cache_clear()
            section, params, up = cfg.stage_signature("diss_combo", base)
            return (
                cfg.compute_hash(section, params, upstream=up),
                resolve._strip_volatile(json.loads(cfg.canonicalize(section, params, up))),
            )

        h_old, sig_old = _sig("OLD")
        h_new, sig_new = _sig("NEW")
        cfg.engine_fingerprint.cache_clear()
        assert h_old != h_new          # code-aware dirs (recompute on code change)
        assert sig_old == sig_new      # plotting matches regardless of code version


# ---------------------------------------------------------------------------
# Part B — per-file cache key + markers
# ---------------------------------------------------------------------------

class TestCacheKeyAndMarker:
    def test_file_fingerprint_missing_file(self, tmp_path):
        assert pl._file_fingerprint(tmp_path / "nope.p") == {"missing": True}

    def test_cachekey_changes_with_input(self, tmp_path):
        p = tmp_path / "a.p"
        p.write_bytes(b"x" * 10)
        k1 = pl._file_cachekey(p, {}, {"diss": "h1"})
        p.write_bytes(b"y" * 20)  # different size
        k2 = pl._file_cachekey(p, {}, {"diss": "h1"})
        assert k1 != k2

    def test_cachekey_changes_with_stage_hash(self, tmp_path):
        """A config/engine change (different stage signature hash) => new key,
        even with byte-identical input (this is why we key on the hash, not the
        reusable {stage}_NN basename)."""
        p = tmp_path / "a.p"
        p.write_bytes(b"x" * 10)
        assert pl._file_cachekey(p, {}, {"diss": "h1"}) != \
            pl._file_cachekey(p, {}, {"diss": "h2"})

    def test_cachekey_changes_with_external_input(self, tmp_path):
        p = tmp_path / "a.p"
        p.write_bytes(b"x")
        assert pl._file_cachekey(p, {}, {}) != \
            pl._file_cachekey(p, {"hotel": {"size": 1, "mtime_ns": 2}}, {})

    def test_marker_current_requires_matching_key(self, tmp_path):
        m = pl._marker_path(tmp_path, "stem")
        m.parent.mkdir()
        m.write_text(json.dumps({"cachekey": "K", "outputs": []}))
        assert pl._marker_is_current(m, "K", set())
        assert not pl._marker_is_current(m, "DIFFERENT", set())

    def test_marker_current_requires_outputs_present(self, tmp_path):
        """A marker is a claim; the files are the truth. A recorded output absent
        from the present-set forces a recompute (prune-then-readd safety)."""
        m = pl._marker_path(tmp_path, "a")
        m.parent.mkdir()
        m.write_text(json.dumps({"cachekey": "K", "outputs": ["diss_00/a_prof001.nc"]}))
        assert pl._marker_is_current(m, "K", {"diss_00/a_prof001.nc"})
        assert not pl._marker_is_current(m, "K", set())  # output no longer present

    def test_list_present_outputs_one_glob_per_stage(self, tmp_path):
        """The present-set is gathered with one listing per stage dir (so the
        per-file check is membership, not a stat per NC)."""
        d = tmp_path / "diss_00"
        d.mkdir()
        (d / "a_prof001.nc").write_bytes(b"x")
        (d / "b_prof001.nc").write_bytes(b"x")
        (d / "ignore.txt").write_bytes(b"x")
        present = pl._list_present_outputs({"diss": d}, tmp_path)
        assert present == {"diss_00/a_prof001.nc", "diss_00/b_prof001.nc"}

    def test_stage_signature_hashes_reads_sig_files(self, tmp_path):
        d = tmp_path / "diss_00"
        d.mkdir()
        (d / ".params_sha256_abc123").write_text("{}")
        assert pl._stage_signature_hashes({"diss": d}) == {"diss": "abc123"}


# ---------------------------------------------------------------------------
# Part C — bin/combo manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_mtime_immune(self, tmp_path):
        """The manifest keys on per-file cache keys, NOT mtime, so it is stable
        on exFAT (2 s mtime granularity) — touching a file doesn't change it."""
        nc = tmp_path / "a_prof001.nc"
        nc.write_bytes(b"x")
        ck = {"a": "KEY"}
        m1 = pl._inputs_manifest([nc], ck)
        os.utime(nc, (nc.stat().st_atime, nc.stat().st_mtime + 5))  # bump mtime
        m2 = pl._inputs_manifest([nc], ck)
        assert m1 == m2  # mtime change does not affect the manifest

    def test_manifest_changes_with_cachekey(self, tmp_path):
        nc = tmp_path / "a_prof001.nc"
        nc.write_bytes(b"x")
        assert pl._inputs_manifest([nc], {"a": "K1"}) != \
            pl._inputs_manifest([nc], {"a": "K2"})

    def test_output_is_current_none_manifest(self, tmp_path):
        nc = tmp_path / "binned.nc"
        nc.write_bytes(b"x")
        assert not pl._output_is_current(nc, None)  # None => always rebuild


# ---------------------------------------------------------------------------
# End-to-end: run_pipeline twice -> the second run skips unchanged files
# ---------------------------------------------------------------------------

def _make_combo_noop(*a, **k):
    return None


class TestIncrementalRun:
    def _config(self, tmp_path):
        return {
            "files": {"output_root": str(tmp_path / "out"), "trim": False},
            "profiles": {}, "epsilon": {}, "fp07": {}, "ct": {},
            "ctd": {"enable": False}, "chi": {"enable": False},
        }

    @pytest.fixture
    def _harness(self, tmp_path, monkeypatch):
        import numpy as np
        import xarray as xr

        src = tmp_path / "VMP" / "a.p"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"raw-data")

        calls: list[str] = []

        def fake_process(p_path, config, gps, output_dirs, **kw):
            stem = kw["output_stem"]
            calls.append(stem)
            for stage in ("profiles", "diss"):
                xr.Dataset({"x": ("bin", np.zeros(3))}).to_netcdf(
                    output_dirs[stage] / f"{stem}_prof001.nc"
                )
            return {"source": str(p_path), "profiles": ["p"], "diss": ["d"], "chi": []}

        monkeypatch.setattr(pl, "process_file", fake_process)
        monkeypatch.setattr(pl, "_run_combo", _make_combo_noop)
        monkeypatch.setattr(
            "odas_tpw.perturb.binning.bin_by_depth", lambda *a, **k: xr.Dataset()
        )
        monkeypatch.setattr(
            "odas_tpw.perturb.binning.bin_diss", lambda *a, **k: xr.Dataset()
        )
        return src, calls

    def test_second_run_skips_unchanged_file(self, tmp_path, _harness):
        src, calls = _harness
        config = self._config(tmp_path)
        pl.run_pipeline(config, p_files=[src])
        assert len(calls) == 1                        # processed once
        pl.run_pipeline(config, p_files=[src])
        assert len(calls) == 1                        # second run skipped it

    def test_force_reprocesses(self, tmp_path, _harness):
        src, calls = _harness
        config = self._config(tmp_path)
        pl.run_pipeline(config, p_files=[src])
        config["files"]["force"] = True
        pl.run_pipeline(config, p_files=[src])
        assert len(calls) == 2                        # --force ignores the cache

    def test_changed_input_reprocesses(self, tmp_path, _harness):
        src, calls = _harness
        config = self._config(tmp_path)
        pl.run_pipeline(config, p_files=[src])
        src.write_bytes(b"raw-data-CHANGED-longer")   # different size -> new key
        pl.run_pipeline(config, p_files=[src])
        assert len(calls) == 2

    def test_pruned_output_reprocesses(self, tmp_path, _harness):
        """A marker present but its output deleted must NOT false-skip."""
        src, calls = _harness
        config = self._config(tmp_path)
        pl.run_pipeline(config, p_files=[src])
        out_root = Path(config["files"]["output_root"])
        deleted = [nc for nc in out_root.glob("diss_*/*_prof*.nc")]
        assert deleted  # sanity: there was an output to delete
        for nc in deleted:
            nc.unlink()                               # delete a recorded output
        pl.run_pipeline(config, p_files=[src])
        assert len(calls) == 2

    def test_failed_processing_is_not_cached(self, tmp_path, monkeypatch):
        """A caught failure (process_file returns a partial result with
        ``errors``) must NOT be cached — the next same-config run retries
        instead of locking in incomplete science. A subsequent clean run caches."""
        import numpy as np
        import xarray as xr

        src = tmp_path / "VMP" / "a.p"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"raw-data")

        calls: list[str] = []
        fail = {"v": True}

        def flaky_process(p_path, config, gps, output_dirs, **kw):
            stem = kw["output_stem"]
            calls.append(stem)
            xr.Dataset({"x": ("bin", np.zeros(3))}).to_netcdf(
                output_dirs["profiles"] / f"{stem}_prof001.nc"
            )
            result = {"source": str(p_path), "profiles": ["p"], "diss": [], "chi": []}
            if fail["v"]:                              # simulate a caught diss failure
                result["errors"] = ["diss boom"]
            else:
                xr.Dataset({"x": ("bin", np.zeros(3))}).to_netcdf(
                    output_dirs["diss"] / f"{stem}_prof001.nc"
                )
                result["diss"] = ["d"]
            return result

        monkeypatch.setattr(pl, "process_file", flaky_process)
        monkeypatch.setattr(pl, "_run_combo", _make_combo_noop)
        monkeypatch.setattr(
            "odas_tpw.perturb.binning.bin_by_depth", lambda *a, **k: xr.Dataset()
        )
        monkeypatch.setattr(
            "odas_tpw.perturb.binning.bin_diss", lambda *a, **k: xr.Dataset()
        )

        config = self._config(tmp_path)
        pl.run_pipeline(config, p_files=[src])        # run 1: failed -> not cached
        assert len(calls) == 1
        out_root = Path(config["files"]["output_root"])
        assert not list(out_root.glob(".cache/*.json"))  # no marker for a failed run
        pl.run_pipeline(config, p_files=[src])        # run 2: retries (no skip)
        assert len(calls) == 2

        fail["v"] = False                              # now it succeeds -> caches
        pl.run_pipeline(config, p_files=[src])
        assert len(calls) == 3
        pl.run_pipeline(config, p_files=[src])         # run 4: clean result cached -> skip
        assert len(calls) == 3

    def test_bin_combo_skip_when_inputs_unchanged(self, tmp_path, monkeypatch):
        """Part C end-to-end with NON-empty binned data: a second unchanged run
        skips re-binning AND re-combining (manifest match); a changed input
        forces both to re-run."""
        import numpy as np
        import xarray as xr

        src = tmp_path / "VMP" / "a.p"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"raw-data")

        def fake_process(p_path, config, gps, output_dirs, **kw):
            stem = kw["output_stem"]
            for stage in ("profiles", "diss"):
                xr.Dataset({"x": ("bin", np.zeros(3))}).to_netcdf(
                    output_dirs[stage] / f"{stem}_prof001.nc"
                )
            return {"source": str(p_path), "profiles": ["p"], "diss": ["d"], "chi": []}

        bin_calls, combo_calls = [], []

        def fake_bin(*a, **k):
            bin_calls.append(1)
            return xr.Dataset({"v": ("bin", np.array([1.0, 2.0, 3.0]))})

        def fake_make_combo(src_dir, dst, schema, **k):
            combo_calls.append(1)
            out = Path(dst) / "combo.nc"
            xr.Dataset({"v": ("bin", np.array([1.0, 2.0]))}).to_netcdf(out)
            return out

        monkeypatch.setattr(pl, "process_file", fake_process)
        monkeypatch.setattr("odas_tpw.perturb.binning.bin_by_depth", fake_bin)
        monkeypatch.setattr("odas_tpw.perturb.binning.bin_diss", fake_bin)
        monkeypatch.setattr("odas_tpw.perturb.combo.make_combo", fake_make_combo)

        config = self._config(tmp_path)
        pl.run_pipeline(config, p_files=[src])
        assert len(bin_calls) == 2 and len(combo_calls) == 2   # profiles + diss

        pl.run_pipeline(config, p_files=[src])                  # unchanged
        assert len(bin_calls) == 2 and len(combo_calls) == 2   # both skipped

        src.write_bytes(b"raw-data-CHANGED-longer")             # new input
        pl.run_pipeline(config, p_files=[src])
        assert len(bin_calls) == 4 and len(combo_calls) == 4   # rebin + recombo
