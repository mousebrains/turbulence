# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""
Parser for the pre-v6 ODAS setup-file dialect and setup-file discovery.

ODAS header-version-1 ``.p`` files (pre-2015 instruments, e.g. VMP-2000 SN002,
Taiwan 2013 — GitHub issue #141) carry **no embedded configuration**: record 0
holds the binary address matrix, and the configuration lives in an external
setup file written in the old, pre-INI dialect::

    # whole-line comment
    rate:     512
    recsize:  1
    no-fast:  8
    no-slow:  2
    channel:  10,Pres,4.46,0.049994,1.341e-8,0,...
    matrix:   255 0 1 2 3 5 7 8 9 12
    matrix:   4 6 1 2 3 5 7 8 9 12
    ...

Values are separated by whitespace OR commas (not mixed). The dialect has no
channel ``type`` field and no ``[section]`` headers; sensor type is implied by
the **address** (the old ODAS hardware's fixed channel map, documented in the
setup files' own comment blocks and re-verified against the 2013 ground-truth
products — see ``scratchpad/w7_v1_format/SPEC.md`` on issue #141).

:func:`parse_setup_v1` emits the same dict shape as
:func:`odas_tpw.rsi.p_file.parse_config` (``root`` / ``matrix`` / ``channels``
/ ``instrument_info`` / ``cruise_info``), with channel dicts carrying modern
keys (``id``/``name``/``type``/``coef0``…) so the existing v6 engine converts
them without modification.

Documented dialect EXTENSIONS (keys the 2013 dialect did not have, which users
may add to their setup-file copies):

- ``sh1_sens: 0.0893`` (generic ``<name>_sens:``) — shear-probe sensitivity.
  No original setup file carries sens (probes were swapped mid-cruise; the
  values lived on paper cal sheets), and a shear channel without sens is a
  loud error at conversion time, never a silent default.
- ``model:``, ``sn:``, ``vehicle:`` — instrument identity for
  ``instrument_info`` (the old dialect records none; ``vehicle`` defaults to
  ``vmp`` for a ``profile: vertical`` setup).
"""

import re
import warnings
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

__all__ = [
    "discover_setup_candidates",
    "load_setup_file",
    "parse_setup_v1",
]

# ---------------------------------------------------------------------------
# The old ODAS address map: address -> (canonical channel name, sensor type).
#
# Sources: the setup files' own comment blocks (setup.txt:24-47 in the 2013
# corpus), the ODAS Matlab Library Manual v4.01 App. A, and channel-level
# verification against the 2013 TAI_013_NNN.mat products (SPEC §2.4, issue
# #141). "therm" addresses are emitted as RAW COUNTS (policy: no
# non-authoritative thermistor coefficients — see issue #141); "sbt"/"sbc"
# are Sea-Bird SBE3/SBE4 even/odd 32-bit period-count pairs.
# ---------------------------------------------------------------------------

_V1_ADDRESSES: dict[int, tuple[str, str]] = {
    0: ("Gnd", "gnd"),
    1: ("Ax", "accel"),
    2: ("Ay", "accel"),
    3: ("Az", "accel"),
    4: ("T1", "therm"),
    5: ("T1_dT1", "therm"),
    6: ("T2", "therm"),
    7: ("T2_dT2", "therm"),
    8: ("sh1", "shear"),
    9: ("sh2", "shear"),
    10: ("P", "poly"),
    11: ("P_dP", "poly"),
    12: ("C1_dC1", "raw"),
    16: ("SBT1E", "sbt"),
    17: ("SBT1O", "sbt"),
    18: ("SBC1E", "sbc"),
    19: ("SBC1O", "sbc"),
    255: ("sp_char", "raw"),
}

# Even/odd 32-bit word pairs (even = least-significant 16 bits).
_V1_PAIRS: dict[int, int] = {16: 17, 18: 19}
_V1_PAIR_ODDS = set(_V1_PAIRS.values())

# Shear conversion constants verified against the 2013 products (SPEC §2.4-5):
# counts * (5.0 / 2^16) / (2*sqrt(2) * sens) with electronics gain 1.
_V1_SHEAR_PARAMS = {"adc_fs": "5.0", "adc_bits": "16", "diff_gain": "1.0"}

_KEY_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*(.*)$")

# Instrument-identity extension keys routed to instrument_info.
_INSTRUMENT_KEYS = ("model", "sn", "vehicle")


def _split_values(raw: str) -> list[str]:
    """Split a value list on commas OR whitespace (the dialect forbids mixing)."""
    if "," in raw:
        return [v.strip() for v in raw.split(",")]
    return raw.split()


def _strip_trailing_zeros(values: list[str]) -> list[str]:
    """Drop trailing all-zero padding values, keeping at least one value."""
    end = len(values)
    while end > 1:
        try:
            if float(values[end - 1]) == 0.0:
                end -= 1
                continue
        except ValueError:
            pass
        break
    return values[:end]


def _merged_pair_name(line_name: str, addr_even: int) -> str:
    """Channel name for a merged even/odd pair: strip the E/O suffix.

    ``SBT1E``/``SBT1O`` -> ``SBT1``. Falls back to the canonical table name
    without its suffix when the setup file has no channel line.
    """
    if line_name and line_name[-1] in ("E", "e", "O", "o"):
        return line_name[:-1]
    return line_name or _V1_ADDRESSES[addr_even][0][:-1]


def parse_setup_v1(
    text: str,
    *,
    sens_overrides: dict[str, float] | None = None,
    source: str = "setup",
) -> dict[str, Any]:
    """Parse an old-dialect setup file into the ``parse_config`` dict shape.

    Parameters
    ----------
    text : str
        Setup-file content (old ``key: values`` dialect).
    sens_overrides : dict, optional
        Shear sensitivities by channel name (case-insensitive, e.g.
        ``{"sh1": 0.0893}``); takes precedence over ``<name>_sens:``
        extension keys in the file.
    source : str
        Label used in warnings (usually the setup-file name).

    Returns
    -------
    dict with the same shape as :func:`odas_tpw.rsi.p_file.parse_config`:
    ``{"matrix": [...], "channels": [...], "instrument_info": {...},
    "cruise_info": {}, "root": {...}}``. Channel dicts use modern keys, so
    ``channels.py`` converters apply unchanged.
    """
    sens_map = {k.lower(): v for k, v in (sens_overrides or {}).items()}

    root: dict[str, str] = {}
    matrix: list[list[int]] = []
    # addr -> (name, [value strings]) from 'channel:' lines
    channel_lines: dict[int, tuple[str, list[str]]] = {}
    sens_ext: dict[str, str] = {}  # lower-cased channel name -> sens string
    instrument_info: dict[str, str] = {}

    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _KEY_LINE_RE.match(line)
        if m is None:
            warnings.warn(f"{source}:{lineno}: unrecognized line skipped: {line!r}")
            continue
        key = m.group(1).lower()
        raw_val = m.group(2).strip()

        if key == "matrix":
            try:
                matrix.append([int(v) for v in _split_values(raw_val)])
            except ValueError as exc:
                raise ValueError(f"{source}:{lineno}: malformed matrix row: {raw_val!r}") from exc
        elif key == "channel":
            values = _split_values(raw_val)
            if len(values) < 2:
                raise ValueError(f"{source}:{lineno}: malformed channel line: {raw_val!r}")
            try:
                addr = int(values[0])
            except ValueError as exc:
                raise ValueError(
                    f"{source}:{lineno}: non-numeric channel address: {values[0]!r}"
                ) from exc
            name = values[1]
            coeffs = values[2:]
            if addr in channel_lines:
                warnings.warn(
                    f"{source}: duplicate 'channel:' line for address {addr}; "
                    "keeping the first"
                )
            else:
                channel_lines[addr] = (name, coeffs)
        elif key.endswith("_sens") and len(key) > 5:
            # Documented dialect extension: sh1_sens: 0.0893
            sens_ext[key[: -len("_sens")]] = raw_val
        elif key in _INSTRUMENT_KEYS:
            # Documented dialect extension: instrument identity.
            instrument_info[key] = raw_val
        else:
            root[key] = raw_val

    if not matrix:
        raise ValueError(f"{source}: no 'matrix:' rows found; not a v1 setup file?")

    addresses = sorted({a for row in matrix for a in row} | set(channel_lines))

    channels: list[dict[str, str]] = []
    consumed_odd: set[int] = set()

    def _sens_for(ch_name: str) -> str | None:
        low = ch_name.lower()
        if low in sens_map:
            return repr(float(sens_map[low]))
        if low in sens_ext:
            return sens_ext[low]
        return None

    for addr in addresses:
        if addr in consumed_odd:
            continue
        canon_name, ch_type = _V1_ADDRESSES.get(addr, (f"ch{addr}", "raw"))
        if addr not in _V1_ADDRESSES:
            warnings.warn(
                f"{source}: matrix address {addr} is not in the documented v1 "
                f"address map; keeping raw counts as channel '{canon_name}'"
            )
        cline = channel_lines.get(addr)

        if addr in _V1_PAIRS:
            # Sea-Bird even/odd 32-bit pair -> one merged channel.
            odd = _V1_PAIRS[addr]
            consumed_odd.add(odd)
            odd_line = channel_lines.get(odd)
            if cline is None and odd_line is None:
                warnings.warn(
                    f"{source}: Sea-Bird pair {addr}/{odd} has no 'channel:' "
                    "coefficients; keeping raw counts"
                )
                channels.append({"id": str(addr), "name": canon_name, "type": "raw"})
                channels.append(
                    {"id": str(odd), "name": _V1_ADDRESSES[odd][0], "type": "raw"}
                )
                continue
            use = cline or odd_line
            assert use is not None
            if cline is not None and odd_line is not None and cline[1] != odd_line[1]:
                warnings.warn(
                    f"{source}: even/odd coefficient mismatch for addresses "
                    f"{addr}/{odd}; using the even line"
                )
            pair_name, coeffs = use
            if len(coeffs) < 7:
                raise ValueError(
                    f"{source}: Sea-Bird channel at address {addr} needs 7 "
                    f"coefficients (got {len(coeffs)}): {coeffs!r}"
                )
            ch: dict[str, str] = {
                "id": f"{addr},{odd}",
                "name": _merged_pair_name(pair_name, addr),
                "type": ch_type,
            }
            # Old positional order maps directly onto modern coef0..coef6
            # (sbt: g,h,i,j,f0,f_ref,n_periods; sbc: g,0,h,i,j,f_ref,
            # n_periods — SPEC §2.4 items 3-4, vendor odas_sbt/sbc_internal).
            for i in range(7):
                ch[f"coef{i}"] = coeffs[i]
            ch["units"] = "[C]" if ch_type == "sbt" else "[mS/cm]"
            channels.append(ch)
            continue

        if addr in _V1_PAIR_ODDS:
            # Odd half present without its even partner in this loop order
            # (even address absent from matrix and lines).
            warnings.warn(
                f"{source}: odd pair address {addr} appears without its even "
                "partner; keeping raw counts"
            )
            channels.append({"id": str(addr), "name": canon_name, "type": "raw"})
            continue

        if ch_type == "gnd":
            channels.append({"id": str(addr), "name": canon_name, "type": "gnd"})
        elif ch_type == "accel":
            if cline is None:
                warnings.warn(
                    f"{source}: accelerometer address {addr} ({canon_name}) has "
                    "no 'channel:' coefficients; keeping raw counts"
                )
                channels.append({"id": str(addr), "name": canon_name, "type": "raw"})
                continue
            _, coeffs = cline
            if len(coeffs) < 2:
                raise ValueError(
                    f"{source}: accelerometer at address {addr} needs 2 "
                    f"coefficients (a, b); got {coeffs!r}"
                )
            # A[m/s^2] = 9.81*(n - a)/b == modern convert_accel with
            # coef0=a, coef1=b and its legacy defaults adc_fs=1, adc_bits=0
            # (SPEC §2.4-2; residual <= 5e-14 vs the 2013 products).
            channels.append(
                {
                    "id": str(addr),
                    "name": canon_name,
                    "type": "accel",
                    "coef0": coeffs[0],
                    "coef1": coeffs[1],
                }
            )
        elif ch_type == "therm":
            # Policy (issue #141): v1 thermistors stay RAW COUNTS. No
            # acquisition setup file carries FP07 coefficients (the lines are
            # commented out), and the only on-disk source reproduces the 2013
            # response shape ~4.2 degC off its in-situ-calibrated product —
            # not trustworthy enough to publish as physical units.
            if cline is not None and _strip_trailing_zeros(cline[1]) not in ([], ["0"]):
                warnings.warn(
                    f"{source}: thermistor coefficients at address {addr} "
                    f"({canon_name}) are ignored; v1 thermistors are kept as "
                    "raw counts (no authoritative calibration; issue #141)"
                )
            channels.append({"id": str(addr), "name": canon_name, "type": "raw"})
        elif ch_type == "shear":
            if cline is not None and _strip_trailing_zeros(cline[1]) not in ([], ["0"]):
                warnings.warn(
                    f"{source}: shear coefficients at address {addr} "
                    f"({canon_name}) are ignored; the v1 shear conversion uses "
                    "the verified 5-V/gain-1 constants plus an explicit sens "
                    "(SPEC §2.4-5, issue #141)"
                )
            ch = {"id": str(addr), "name": canon_name, "type": "shear", **_V1_SHEAR_PARAMS}
            sens = _sens_for(canon_name)
            if sens is not None:
                ch["sens"] = sens
            channels.append(ch)
        elif ch_type == "poly":
            if cline is None:
                warnings.warn(
                    f"{source}: polynomial address {addr} ({canon_name}) has no "
                    "'channel:' coefficients; keeping raw counts"
                )
                channels.append({"id": str(addr), "name": canon_name, "type": "raw"})
                continue
            _, coeffs = cline
            ch = {"id": str(addr), "name": canon_name, "type": "poly"}
            for i, c in enumerate(_strip_trailing_zeros(coeffs)):
                ch[f"coef{i}"] = c
            if addr in (10, 11):
                ch["units"] = "[dBar]"
            channels.append(ch)
        else:  # raw
            channels.append({"id": str(addr), "name": canon_name, "type": "raw"})

    shear_names = sorted(c["name"] for c in channels if c["type"] == "shear")
    unmatched_sens = set(sens_map) - {n.lower() for n in shear_names}
    if unmatched_sens:
        raise ValueError(
            f"{source}: sens override(s) for non-shear/unknown channel(s): "
            f"{', '.join(sorted(unmatched_sens))} (shear channels here: "
            f"{', '.join(shear_names) or 'none'})"
        )

    if "vehicle" not in instrument_info:
        # The old chain's v1 instruments were VMPs (profile: vertical); the
        # dialect records no identity — see the docs for the model/sn/vehicle
        # extension keys.
        instrument_info["vehicle"] = "vmp"

    return {
        "matrix": matrix,
        "channels": channels,
        "instrument_info": instrument_info,
        "cruise_info": {},
        "root": root,
    }


# ---------------------------------------------------------------------------
# Setup-file discovery (issue #141 / plan delta F7)
# ---------------------------------------------------------------------------

# Precedence order. 'setup.txt' first: for the known 2013 corpus the
# acquisition log proves setup.txt (= SetUp2013.txt) drove acquisition, while
# the sibling .cfg files carry a WRONG pressure polynomial and matrix.
_SETUP_NAME_PATTERNS = ("setup.txt", "setup*.txt", "setup*.cfg")


def discover_setup_candidates(p_path: str | Path) -> list[Path]:
    """Sibling setup-file candidates for a v1 ``.p`` file, in precedence order.

    Searches the ``.p`` file's directory, then one level up. Within each
    directory candidates are ranked by name pattern (``setup.txt`` exact,
    then ``setup*.txt``, then ``setup*.cfg``; matching is case-insensitive)
    and alphabetically within a pattern. Case-insensitive duplicates (APFS)
    are dropped deterministically (first pattern/name wins).
    """
    p_path = Path(p_path)
    out: list[Path] = []
    seen: set[tuple[str, str]] = set()
    parent = p_path.resolve().parent
    for d in (parent, parent.parent):
        try:
            names = sorted(x.name for x in d.iterdir() if x.is_file())
        except OSError:
            continue
        for pattern in _SETUP_NAME_PATTERNS:
            for name in names:
                low = name.lower()
                key = (str(d).lower(), low)
                if fnmatch(low, pattern) and key not in seen:
                    seen.add(key)
                    out.append(d / name)
    return out


def load_setup_file(
    path: str | Path,
    *,
    sens_overrides: dict[str, float] | None = None,
) -> tuple[dict[str, Any], str]:
    """Load a setup file of either dialect; returns ``(config_dict, dialect)``.

    Dialect sniffing (plan delta F7): a file containing ``[section]`` headers
    is the INI dialect and routes to :func:`parse_config` (e.g. the
    library-3.1-era ``SETUP.CFG`` files); otherwise it must parse as the old
    ``key: values`` dialect. ``dialect`` is ``"ini"`` or ``"v1"``.
    """
    path = Path(path)
    text = path.read_text(encoding="ascii", errors="replace")
    if re.search(r"^\s*\[.+\]\s*$", text, flags=re.MULTILINE):
        from odas_tpw.rsi.p_file import parse_config

        cfg = parse_config(text)
        if sens_overrides:
            lowmap = {k.lower(): v for k, v in sens_overrides.items()}
            for ch in cfg.get("channels", []):
                sens = lowmap.get(ch.get("name", "").lower())
                if sens is not None:
                    ch["sens"] = repr(float(sens))
        if not cfg.get("matrix"):
            raise ValueError(f"{path.name}: INI-dialect setup file has no [matrix] rows")
        return cfg, "ini"
    return parse_setup_v1(text, sens_overrides=sens_overrides, source=path.name), "v1"
