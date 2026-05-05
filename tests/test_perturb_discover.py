# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.discover — .p file discovery."""

from odas_tpw.perturb.discover import find_p_files


class TestFindPFiles:
    def test_finds_p_files(self, tmp_path):
        (tmp_path / "file1.p").write_bytes(b"\x00")
        (tmp_path / "file2.p").write_bytes(b"\x00")
        result = find_p_files(tmp_path)
        assert len(result) == 2

    def test_excludes_original(self, tmp_path):
        (tmp_path / "file1.p").write_bytes(b"\x00")
        (tmp_path / "file1_original.p").write_bytes(b"\x00")
        result = find_p_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "file1.p"

    def test_excludes_dotfiles(self, tmp_path):
        (tmp_path / "file1.p").write_bytes(b"\x00")
        (tmp_path / ".hidden.p").write_bytes(b"\x00")
        result = find_p_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "file1.p"

    def test_case_insensitive_extension(self, tmp_path):
        (tmp_path / "file1.P").write_bytes(b"\x00")
        result = find_p_files(tmp_path, pattern="*.[pP]")
        assert len(result) == 1

    def test_sorted_output(self, tmp_path):
        (tmp_path / "b.p").write_bytes(b"\x00")
        (tmp_path / "a.p").write_bytes(b"\x00")
        result = find_p_files(tmp_path)
        assert result[0].name == "a.p"
        assert result[1].name == "b.p"

    def test_empty_directory(self, tmp_path):
        assert find_p_files(tmp_path) == []

    def test_subdirectory(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "file1.p").write_bytes(b"\x00")
        result = find_p_files(tmp_path)
        assert len(result) == 1

    def test_non_p_files_excluded(self, tmp_path):
        (tmp_path / "file1.p").write_bytes(b"\x00")
        (tmp_path / "file2.nc").write_bytes(b"\x00")
        (tmp_path / "file3.txt").write_bytes(b"\x00")
        result = find_p_files(tmp_path)
        assert len(result) == 1

    def test_directory_matching_glob_skipped(self, tmp_path):
        """A directory that matches the glob pattern is filtered out."""
        # Create a directory whose name ends in .p — broad pattern matches it
        (tmp_path / "fake.p").mkdir()
        (tmp_path / "real.p").write_bytes(b"\x00")
        result = find_p_files(tmp_path, pattern="*.p")
        # Only the real file is returned; the directory is rejected at is_file()
        assert len(result) == 1
        assert result[0].name == "real.p"

    def test_broad_glob_filters_non_p_files(self, tmp_path):
        """Broad glob ('*') reaches the suffix check (line 34)."""
        (tmp_path / "file1.p").write_bytes(b"\x00")
        (tmp_path / "file2.nc").write_bytes(b"\x00")
        (tmp_path / "file3.txt").write_bytes(b"\x00")
        result = find_p_files(tmp_path, pattern="*")
        assert len(result) == 1
        assert result[0].name == "file1.p"
