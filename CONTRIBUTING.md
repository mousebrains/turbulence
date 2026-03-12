# Contributing to rsi-python

## Development Setup

```bash
git clone https://github.com/mousebrains/rsi-python.git
cd rsi-python
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
mypy src/rsi_python/                # type checking
```

## Pull Request Process

1. Create a feature branch from `main`.
2. Ensure `ruff check`, `mypy`, and `pytest` pass.
3. Add tests for new functionality.
4. Open a PR against `main` with a clear description.

## Style

- Follow existing code conventions (ruff enforces formatting).
- Add docstrings with Parameters/Returns sections for public functions.
- Use NumPy-style docstrings.
- Include unit citations for physical constants and algorithm references.
