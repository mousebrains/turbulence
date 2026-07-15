# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for optional shell tab-completion wiring (argcomplete).

These cover the small shared helper and confirm every CLI actually invokes it
immediately before parsing. They do not exercise the shell completion protocol
itself (that is argcomplete's own responsibility and its wire format varies by
version); a manual smoke test of `rsi-tpw <TAB>` covers that end to end.
"""

import argparse
import sys

import pytest

from odas_tpw import _completion


def _dummy_parser():
    p = argparse.ArgumentParser(prog="dummy")
    p.add_argument("x", nargs="?")
    return p


def test_enable_argcomplete_noop_without_package(monkeypatch):
    """When argcomplete is not importable, the helper is a silent no-op."""
    # Setting the module to None makes `import argcomplete` raise ImportError.
    monkeypatch.setitem(sys.modules, "argcomplete", None)
    _completion.enable_argcomplete(_dummy_parser())  # must not raise


def test_enable_argcomplete_invokes_autocomplete_when_present(monkeypatch):
    """When argcomplete is installed, the parser is handed to autocomplete()."""
    seen = {}

    class _FakeArgcomplete:
        def autocomplete(self, parser, **kwargs):
            seen["parser"] = parser

    monkeypatch.setitem(sys.modules, "argcomplete", _FakeArgcomplete())
    parser = _dummy_parser()
    _completion.enable_argcomplete(parser)
    assert seen["parser"] is parser


class _HookRan(Exception):
    """Sentinel raised by the patched hook to prove it ran before parse_args."""


# (import path, main callable) for every CLI wired for completion.
_CLI_MAINS = [
    "odas_tpw.rsi.cli",
    "odas_tpw.perturb.cli",
    "odas_tpw.perturb.plot.cli",
    "odas_tpw.perturb.diag.cli",
]


@pytest.mark.parametrize("module_path", _CLI_MAINS)
def test_each_cli_calls_enable_argcomplete_before_parsing(module_path, monkeypatch):
    """Each CLI's main() runs enable_argcomplete() ahead of parse_args().

    The CLIs do `from odas_tpw._completion import enable_argcomplete` inside
    main(), so patching the attribute here is picked up at call time. The hook
    raises before parse_args, so no subcommand work runs and argv is irrelevant.
    """
    import importlib

    main = importlib.import_module(module_path).main

    def _boom(_parser):
        raise _HookRan

    monkeypatch.setattr(_completion, "enable_argcomplete", _boom)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(_HookRan):
        main()
