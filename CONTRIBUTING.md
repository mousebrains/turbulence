# Contributing to microstructure-tpw

## Development Setup

```bash
git clone https://github.com/mousebrains/turbulence.git
cd turbulence
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest                    # all tests
python -m pytest tests/test_chi.py  # single file
python -m pytest -n auto            # parallel (requires VMP data for MATLAB tests)
```

## Code Quality

```bash
ruff check src/ tests/              # linting
mypy src/odas_tpw/  # type checking
```

## Coverage policy

CI measures branch coverage on Python 3.14 / Ubuntu and gates merges with
`--cov-fail-under` (current floor in `[tool.coverage.report] fail_under` of
`pyproject.toml`).  Run locally with:

```bash
python -m pytest --cov           # uses pyproject config (branch coverage)
python -m coverage report -m     # see missing lines
```

The following are deliberately omitted from coverage measurement (see
`[tool.coverage.run] omit` in `pyproject.toml`):

| Path | Reason |
|------|--------|
| `src/odas_tpw/rsi/viewer_base.py` | Interactive matplotlib base class (`Button` widgets, draw callbacks). Manual UI testing only. |
| `src/odas_tpw/rsi/quick_look.py` | Interactive profile viewer subclass. |
| `src/odas_tpw/rsi/diss_look.py` | Interactive dissipation viewer subclass. |
| `*/__main__.py` | Module entry points: re-dispatch to `.cli:main`, nothing testable. |

Lines matching `pragma: no cover`, `if __name__ == "__main__":`,
`if TYPE_CHECKING:`, `raise NotImplementedError`, `@abstractmethod`, or
`raise AssertionError` are also excluded.

When adding new interactive UI code, place it inside the omitted modules
or annotate non-pure callbacks with `# pragma: no cover`.  Pure data-prep
helpers used by viewers should live in non-viewer modules and be tested
normally.

## Pull Request Process

1. Create a feature branch from `main`.
2. Ensure `ruff check`, `mypy`, `pytest`, and the coverage gate pass.
3. Add tests for new functionality.
4. Open a PR against `main` with a clear description.

## Style

- Follow existing code conventions (ruff enforces formatting).
- Add docstrings with Parameters/Returns sections for public functions.
- Use NumPy-style docstrings.
- Include unit citations for physical constants and algorithm references.
