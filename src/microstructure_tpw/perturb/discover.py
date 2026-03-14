# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""File discovery for .p files.

Reference: Code/find_P_filenames.m
"""

from pathlib import Path


def find_p_files(root: str | Path, pattern: str = "**/*.p") -> list[Path]:
    """Find .p files under *root* matching *pattern*.

    Filters out ``_original.p`` files and dotfiles (hidden files).
    Returns a sorted list of paths.

    Parameters
    ----------
    root : str or Path
        Directory to search.
    pattern : str
        Glob pattern (default ``"**/*.p"``).

    Returns
    -------
    list of Path
        Sorted list of matching .p file paths.
    """
    root = Path(root)
    results = []
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".p":
            continue
        if p.name.startswith("."):
            continue
        if p.stem.endswith("_original"):
            continue
        results.append(p)
    return sorted(results)
