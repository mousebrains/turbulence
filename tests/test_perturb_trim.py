# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.trim — corrupt record trimming."""

import struct

import pytest

from odas_tpw.perturb.trim import trim_p_file
from odas_tpw.rsi.p_file import _H, HEADER_BYTES, HEADER_WORDS


def _make_p_file(path, *, record_size=1024, config_size=256, n_records=3, extra_bytes=0):
    """Create a synthetic .p file with the given structure.

    Layout: [header_record] [data_record]*n_records [extra_bytes]
    header_record = 128-byte header + config_size bytes of config
    data_record = 128-byte header + (record_size - 128) bytes of data
    """
    header_size = HEADER_BYTES

    # Build a header with the fields trim.py reads
    words = [0] * HEADER_WORDS
    words[_H["header_size"]] = header_size
    words[_H["config_size"]] = config_size
    words[_H["record_size"]] = record_size
    words[_H["endian"]] = 1  # little-endian flag

    hdr_bytes = struct.pack(f"<{HEADER_WORDS}H", *words)
    config_bytes = b"\x00" * config_size

    # First record (header + config)
    data = hdr_bytes + config_bytes

    # Data records (each = record_size bytes)
    for _ in range(n_records):
        data += b"\x01" * record_size

    # Fractional record
    if extra_bytes > 0:
        data += b"\xff" * extra_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class TestTrimPFile:
    def test_complete_file_is_referenced_in_place(self, tmp_path):
        src = _make_p_file(tmp_path / "clean.p", n_records=3, extra_bytes=0)
        out_dir = tmp_path / "trimmed"
        result = trim_p_file(src, out_dir)
        # Complete file: the original is referenced, not copied.
        assert result.action == "referenced"
        assert result.dest == src
        assert result.bytes_removed == 0
        # No output written under the trim directory.
        assert not out_dir.exists()

    def test_trims_fractional_record(self, tmp_path):
        record_size = 1024
        src = _make_p_file(
            tmp_path / "frac.p",
            record_size=record_size,
            n_records=3,
            extra_bytes=50,
        )
        original_size = src.stat().st_size
        out_dir = tmp_path / "trimmed"
        result = trim_p_file(src, out_dir)
        assert result.dest.exists()
        assert result.dest.stat().st_size == original_size - 50
        assert result.action == "trimmed"
        assert result.bytes_removed == 50
        assert result.record_size == record_size

    def test_output_in_correct_directory(self, tmp_path):
        # Incomplete file so the trimmed output is written under out_dir.
        src = _make_p_file(tmp_path / "test.p", extra_bytes=20)
        out_dir = tmp_path / "output"
        result = trim_p_file(src, out_dir)
        assert result.action == "trimmed"
        assert result.dest.parent == out_dir
        assert result.dest.name == "test.p"

    def test_creates_output_dir(self, tmp_path):
        # Incomplete file so trimming materializes the nested output directory.
        src = _make_p_file(tmp_path / "test.p", extra_bytes=20)
        out_dir = tmp_path / "deep" / "nested" / "dir"
        result = trim_p_file(src, out_dir)
        assert result.dest.exists()
        assert result.dest.parent == out_dir

    def test_preserves_complete_records(self, tmp_path):
        record_size = 512
        config_size = 128
        n_records = 5
        src = _make_p_file(
            tmp_path / "multi.p",
            record_size=record_size,
            config_size=config_size,
            n_records=n_records,
            extra_bytes=100,
        )
        out_dir = tmp_path / "trimmed"
        result = trim_p_file(src, out_dir)

        header_size = HEADER_BYTES
        first_record = header_size + config_size
        expected = first_record + n_records * record_size
        assert result.dest.stat().st_size == expected

    def test_file_too_small_raises(self, tmp_path):
        src = tmp_path / "tiny.p"
        src.write_bytes(b"\x00" * 10)
        with pytest.raises(ValueError, match="too small"):
            trim_p_file(src, tmp_path / "out")

    def test_file_smaller_than_first_record_raises(self, tmp_path):
        """File has a header that claims a large config_size, but is truncated."""
        # Build a 128-byte header that says config_size = 10000, then EOF
        words = [0] * HEADER_WORDS
        words[_H["header_size"]] = HEADER_BYTES
        words[_H["config_size"]] = 10000  # absurdly large
        words[_H["record_size"]] = 1024
        words[_H["endian"]] = 1

        src = tmp_path / "truncated.p"
        src.write_bytes(struct.pack(f"<{HEADER_WORDS}H", *words))  # only 128 bytes
        with pytest.raises(ValueError, match="smaller than first record"):
            trim_p_file(src, tmp_path / "out")

    def test_skips_after_a_real_trim(self, tmp_path):
        """A trimmed output is reused on the next run (only trimmed files skip)."""
        src = _make_p_file(
            tmp_path / "frac.p", record_size=1024, n_records=3, extra_bytes=50
        )
        out_dir = tmp_path / "trimmed"

        first = trim_p_file(src, out_dir)
        assert first.action == "trimmed"

        second = trim_p_file(src, out_dir)
        assert second.action == "skipped"
        assert second.dest == first.dest

    def test_force_re_trims_up_to_date_output(self, tmp_path):
        """force=True redoes the trim even when the output is current."""
        src = _make_p_file(
            tmp_path / "frac.p", record_size=1024, n_records=3, extra_bytes=50
        )
        out_dir = tmp_path / "trimmed"

        trim_p_file(src, out_dir)
        forced = trim_p_file(src, out_dir, force=True)
        assert forced.action == "trimmed"

    def test_dest_equals_source_does_not_destroy_file(self, tmp_path):
        """When the trim dest resolves to the source (root==output_dir==source
        dir), the source must NOT be truncated to 0 bytes — it's produced via a
        temp file + atomic replace (M-10)."""
        src = _make_p_file(
            tmp_path / "frac.p", record_size=1024, n_records=3, extra_bytes=50
        )
        original_size = src.stat().st_size
        result = trim_p_file(src, tmp_path, root=tmp_path)  # dest resolves to src
        assert result.action == "trimmed"
        assert result.dest.resolve() == src.resolve()
        # Source not wiped; it's the trimmed prefix (the 50 fractional bytes gone).
        assert src.stat().st_size == original_size - 50

    def test_resaves_when_source_is_newer(self, tmp_path):
        """A source modified after its trimmed output is re-trimmed, not skipped."""
        import os

        src = _make_p_file(
            tmp_path / "frac.p", record_size=1024, n_records=3, extra_bytes=50
        )
        out_dir = tmp_path / "trimmed"

        first = trim_p_file(src, out_dir)
        # Bump the source mtime one second past the existing trimmed output.
        newer = first.dest.stat().st_mtime_ns + 1_000_000_000
        os.utime(src, ns=(newer, newer))

        second = trim_p_file(src, out_dir)
        assert second.action == "trimmed"

    def test_repaired_complete_source_not_skipped_despite_older_mtime(self, tmp_path):
        """An incomplete source that is later repaired to completeness must not
        keep serving its stale truncated trim, even when the repaired source
        carries an equal/older mtime (backup/cp -p/rsync scenario, audit #68)."""
        import os

        src = tmp_path / "frac.p"
        _make_p_file(src, record_size=1024, n_records=3, extra_bytes=50)
        out_dir = tmp_path / "trimmed"

        first = trim_p_file(src, out_dir)
        assert first.action == "trimmed"
        dest_mtime = first.dest.stat().st_mtime_ns

        # Repair the source to a complete file (no fractional record) but stamp
        # it with an OLDER mtime than the trimmed output.
        _make_p_file(src, record_size=1024, n_records=4, extra_bytes=0)
        older = dest_mtime - 1_000_000_000
        os.utime(src, ns=(older, older))

        second = trim_p_file(src, out_dir)
        # Now complete -> referenced to the repaired source, not the stale trim.
        assert second.action == "referenced"
        assert second.dest == src

    def test_skip_requires_size_match_not_only_mtime(self, tmp_path):
        """A geometry change that alters the trimmed size must re-trim even with
        an older source mtime (size check, not mtime alone, audit #68)."""
        import os

        src = tmp_path / "frac.p"
        _make_p_file(src, record_size=1024, n_records=3, extra_bytes=50)
        out_dir = tmp_path / "trimmed"
        first = trim_p_file(src, out_dir)
        dest_mtime = first.dest.stat().st_mtime_ns

        # Replace with a still-incomplete source but a different trimmed size
        # (fewer complete records), older mtime.
        _make_p_file(src, record_size=1024, n_records=2, extra_bytes=50)
        older = dest_mtime - 1_000_000_000
        os.utime(src, ns=(older, older))

        second = trim_p_file(src, out_dir)
        assert second.action == "trimmed"

    def test_invalid_record_size_raises(self, tmp_path):
        """A record_size of 0 is rejected."""
        words = [0] * HEADER_WORDS
        words[_H["header_size"]] = HEADER_BYTES
        words[_H["config_size"]] = 16
        words[_H["record_size"]] = 0  # invalid
        words[_H["endian"]] = 1

        src = tmp_path / "bad_record.p"
        # 128 byte header + 16 bytes config = 144 bytes total, file_size >= first_record_size
        src.write_bytes(struct.pack(f"<{HEADER_WORDS}H", *words) + b"\x00" * 16)
        with pytest.raises(ValueError, match="invalid record_size"):
            trim_p_file(src, tmp_path / "out")


class TestTrimCache:
    """The incremental trim-decision cache (skips the per-file header read on an
    unchanged source — the cost that dominates over a slow SMB mount)."""

    def test_cache_hit_skips_header_read(self, tmp_path, monkeypatch):
        src = _make_p_file(tmp_path / "vmp" / "clean.p", n_records=3)
        out_dir = tmp_path / "trimmed"
        cache = tmp_path / ".cache"

        r1 = trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache)
        assert r1.action == "referenced"
        assert list((cache / "trim").rglob("*.json"))  # marker written

        # Make the header-read path explode; a cache hit must not reach it.
        import odas_tpw.perturb.trim as trim_mod
        def _boom(*a, **k):
            raise AssertionError("header was read on a cache hit")
        monkeypatch.setattr(trim_mod, "_detect_endian", _boom)

        r2 = trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache)
        assert r2.action == "referenced" and r2.dest == src   # served from cache

        with pytest.raises(AssertionError):                    # --force bypasses cache
            trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache, force=True)

    def test_cache_invalidated_when_source_changes(self, tmp_path):
        src = _make_p_file(tmp_path / "vmp" / "f.p", n_records=3)
        out_dir = tmp_path / "trimmed"
        cache = tmp_path / ".cache"
        assert trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache).action == "referenced"

        # Source grows an incomplete final record -> different size -> cache miss
        # -> header re-read -> now trimmed.
        _make_p_file(src, n_records=3, extra_bytes=40)
        r = trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache)
        assert r.action == "trimmed"

    def test_cache_misses_when_trimmed_output_deleted(self, tmp_path):
        src = _make_p_file(tmp_path / "vmp" / "f.p", n_records=3, extra_bytes=40)
        out_dir = tmp_path / "trimmed"
        cache = tmp_path / ".cache"
        r1 = trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache)
        assert r1.action == "trimmed" and r1.dest.exists()

        r1.dest.unlink()  # trimmed output removed -> cache must not claim it current
        r2 = trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache)
        assert r2.action == "trimmed" and r2.dest.exists()  # re-trimmed

    def test_cache_misses_when_trimmed_output_truncated(self, tmp_path):
        """A trimmed output that exists but is the wrong size (truncated/
        corrupted) must not be served from cache as up to date."""
        src = _make_p_file(tmp_path / "vmp" / "f.p", n_records=3, extra_bytes=40)
        out_dir = tmp_path / "trimmed"
        cache = tmp_path / ".cache"
        r1 = trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache)
        assert r1.action == "trimmed"

        with open(r1.dest, "r+b") as f:   # corrupt the trimmed output, source unchanged
            f.truncate(10)
        r2 = trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache)
        assert r2.action == "trimmed" and r2.dest.stat().st_size > 10  # re-trimmed

    def test_malformed_marker_is_a_miss_not_a_crash(self, tmp_path):
        """A corrupt/hand-edited/future-schema marker must degrade to a cache
        miss (fall back to the header read), never abort the trim."""
        import json

        src = _make_p_file(tmp_path / "vmp" / "clean.p", n_records=3)
        out_dir = tmp_path / "trimmed"
        cache = tmp_path / ".cache"
        trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache)   # populate marker
        marker = next((cache / "trim").rglob("*.json"))

        data = json.loads(marker.read_text())
        data["record_size"] = {}                                   # non-scalar -> int() would raise
        marker.write_text(json.dumps(data))
        assert trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache).action == "referenced"

        marker.write_text("[1, 2, 3]")                             # non-dict payload
        assert trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache).action == "referenced"

        marker.write_text("{ not valid json")                      # unparseable
        assert trim_p_file(src, out_dir, root=tmp_path, cache_dir=cache).action == "referenced"
