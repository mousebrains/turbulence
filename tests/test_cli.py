# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the rsi-tpw CLI."""

import sys

import pytest


def test_help_exits_cleanly(monkeypatch):
    """rsi-tpw --help should exit with code 0."""
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "--help"])
    from rsi_python.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
