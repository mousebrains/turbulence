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
    for p in glob_paths(root, pattern):
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


def glob_paths(root: Path, pattern: str) -> list[Path]:
    """``root``/``pattern`` -> paths, **following symlinked directories**.

    ``Path.glob`` does not traverse a symlinked directory with ``**``, and it
    says nothing when it declines to: a deployment whose data is symlinked in
    (``MR -> /Volumes/.../osu685/MR``, which is how a big dataset is normally
    kept off the local disk) matched zero files under the default
    ``**/*.p`` while ``ls MR/*.p`` showed 1228. The stdlib ``glob`` module
    does follow them, and matches what a shell would do -- which is what
    someone writing a glob into a config expects.

    Python 3.13 added ``Path.glob(..., recurse_symlinks=True)``, but 3.12 is
    still supported here, so use ``glob`` for both.
    """
    import glob as globmod

    return [
        Path(root) / rel
        for rel in globmod.glob(pattern, root_dir=str(root), recursive=True)
    ]
