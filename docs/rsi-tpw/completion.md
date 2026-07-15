# Shell tab-completion

The argparse-based CLIs support `<TAB>` completion of subcommands, flags, and
file arguments via [argcomplete](https://github.com/kislyuk/argcomplete). It is
**opt-in**: nothing changes until you install the extra and register the
completion in your shell.

Four commands are wired for completion:

- `rsi-tpw`
- `perturb`
- `perturb-plot`
- `perturb-diag`

## 1. Install argcomplete

```bash
pip install 'microstructure-tpw[completion]'   # adds argcomplete
```

(Completion is entirely optional; the CLIs run identically without it.)

### If you installed with pipx

pipx isolates the app in its own venv, so the extra above will not reach it and
the `register-python-argcomplete` helper is not put on `PATH`. Inject argcomplete
into the app's venv instead:

```bash
pipx inject microstructure-tpw argcomplete
```

The helper then lives at
`~/.local/pipx/venvs/microstructure-tpw/bin/register-python-argcomplete`; use
that full path in step 2 (the pipx block below).

## 2. Register it in your shell

Use **per-command registration** — it names each command explicitly and does not
rely on the global-activation marker (which console-script wrappers do not
carry).

### zsh

The generated hook uses zsh's native `compdef` (from `compinit`) *and* the
bash-compat `complete` (from `bashcompinit`), so **both** must be loaded first —
`compinit` before `bashcompinit`. Missing `compinit` gives
`command not found: compdef`. Add to `~/.zshrc`:

```zsh
autoload -U compinit && compinit
autoload -U bashcompinit && bashcompinit
eval "$(register-python-argcomplete rsi-tpw)"
eval "$(register-python-argcomplete perturb)"
eval "$(register-python-argcomplete perturb-plot)"
eval "$(register-python-argcomplete perturb-diag)"
```

If something else already runs `compinit` (e.g. oh-my-zsh), keep that call
*before* these lines; calling `compinit` twice is harmless.

**pipx variant** — `register-python-argcomplete` is inside the app venv, not on
`PATH`, so call it by full path:

```zsh
autoload -U compinit && compinit
autoload -U bashcompinit && bashcompinit
_mtpw_reg=~/.local/pipx/venvs/microstructure-tpw/bin/register-python-argcomplete
for _cmd in rsi-tpw perturb perturb-plot perturb-diag; do
  eval "$("$_mtpw_reg" "$_cmd")"
done
unset _cmd _mtpw_reg
```

### bash

Add to `~/.bashrc` (bash-completion must be installed and sourced):

```bash
eval "$(register-python-argcomplete rsi-tpw)"
eval "$(register-python-argcomplete perturb)"
eval "$(register-python-argcomplete perturb-plot)"
eval "$(register-python-argcomplete perturb-diag)"
```

Open a new shell (or `source` the rc file) and try:

```text
rsi-tpw <TAB>          # info config cutp nc init patch-config … sensors
rsi-tpw co<TAB>        # -> config
perturb-plot <TAB>     # figure eps-chi overview scalar profiles … gamma-scaling
```

## Notes

- Each `<TAB>` launches Python to introspect the parser, so completion is always
  in sync with the CLI but adds a small (~50–150 ms) latency per keypress. This
  is inherent to argcomplete's dynamic approach.
- Completion is driven by the `_ARGCOMPLETE` environment variable that
  `register-python-argcomplete` sets; ordinary command runs never touch it, so
  there is no overhead or behavior change outside of `<TAB>`.
- The global activation route (`activate-global-python-argcomplete`) is not
  recommended here: it scans the executable for a `# PYTHON_ARGCOMPLETE_OK`
  marker, which the setuptools console-script wrappers for these entry points do
  not contain. Per-command registration above sidesteps that entirely.
