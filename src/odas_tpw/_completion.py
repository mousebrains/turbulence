# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Optional shell tab-completion support (argcomplete).

``argcomplete`` is an optional dependency (the ``completion`` extra). When it is
installed *and* the shell's completion machinery is driving the process,
:func:`enable_argcomplete` lets it emit completions for subcommands, flags, and
file arguments. It is a no-op in every other case — normal runs, and installs
without the extra, are unaffected.

See docs/rsi-tpw/completion.md for how to turn it on in your shell.
"""

from __future__ import annotations

import argparse


def enable_argcomplete(parser: argparse.ArgumentParser) -> None:
    """Attach argcomplete to *parser* if the package is installed.

    Call this immediately before ``parser.parse_args()``. argcomplete only does
    anything when the ``_ARGCOMPLETE`` environment variable is set (which the
    shell completion hook installed by ``register-python-argcomplete`` sets); on
    every ordinary invocation ``autocomplete`` returns immediately, so this adds
    no measurable overhead to real runs and cannot interfere with parsing.
    """
    try:
        import argcomplete
    except ImportError:
        return
    argcomplete.autocomplete(parser)
