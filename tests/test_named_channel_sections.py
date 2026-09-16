# Sep-2026, Claude and Pat Welch, pat@mousebrains.com
"""Old configs that give each channel its own NAMED stanza.

RSI's 2012 MicroRider template (e.g. MR1000-LP SN 046, Taiwan 2013) declares
channels as ``[pitch]``, ``[shear1]``, ``[therm1]``, ... instead of repeated
``[channel]`` stanzas. ``setupstr.m`` never looks at the stanza name -- it finds
channels by ``id`` (or ``id_even``/``id_odd``) and converts those with ``name``
and ``type`` -- so these files convert in ODAS. Before this fix ``parse_config``
accepted only ``[channel]``, and ``PFile`` built ZERO channels from all 77 such
files ("matrix address(es) ... have no usable [channel] section").
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from odas_tpw.rsi import config_patch as cp
from odas_tpw.rsi.p_file import PFile, is_channel_section, parse_config
from tests.test_p_file_branches import _make_minimal_p_file

_MATRIX = """
[matrix]
row1 = 0 1 2 3
row2 = 0 1 2 3
row3 = 0 1 2 3
row4 = 0 1 2 3
"""

# The SN 046 dialect: named stanzas, keys in the order RSI wrote them (note
# [roll] puts type before id, and [shear1] has no units).
_NAMED = (
    """rate=512
profile=horizontal
"""
    + _MATRIX
    + """
[gnd1]
id=0
type=gnd
name=Gnd
coef0=0

[pitch]
id=1
type=accel
name=Ax
coef0=-54.5
coef1=13023.5

[roll]
type=accel
id=2
name=Ay
coef0=-105
coef1=13352

[shear1]
id=3
type=shear
name=sh1
diff_gain=0.97
sens=0.0700
adc_fs=4.096
adc_bits=16
"""
)

# The same channels in the modern dialect.
_MODERN = (
    """rate=512
profile=horizontal
"""
    + _MATRIX
    + """
[channel]
id=0
type=gnd
name=Gnd
coef0=0

[channel]
id=1
type=accel
name=Ax
coef0=-54.5
coef1=13023.5

[channel]
type=accel
id=2
name=Ay
coef0=-105
coef1=13352

[channel]
id=3
type=shear
name=sh1
diff_gain=0.97
sens=0.0700
adc_fs=4.096
adc_bits=16
"""
)


class TestParseConfig:
    def test_named_stanzas_are_channels_in_file_order(self):
        cfg = parse_config(_NAMED)
        assert [c["name"] for c in cfg["channels"]] == ["Gnd", "Ax", "Ay", "sh1"]

    def test_named_matches_modern(self):
        assert parse_config(_NAMED)["channels"] == parse_config(_MODERN)["channels"]

    def test_named_stanza_still_kept_as_a_section(self):
        # setupstr.m keeps every section, so the raw stanza stays under its own
        # name. It is raw text only: corrections go to the channel dict.
        assert parse_config(_NAMED)["shear1"]["sens"] == "0.0700"

    def test_id_after_name_is_still_a_channel(self):
        text = _MATRIX + "\n[therm1]\nname=T1\ntype=therm\nid=0\n"
        assert [c["name"] for c in parse_config(text)["channels"]] == ["T1"]

    def test_id_even_odd_named_stanza_is_a_channel(self):
        text = _MATRIX + "\n[big]\nid_even=1\nid_odd=2\nname=bigval\ntype=raw\n"
        assert [c["name"] for c in parse_config(text)["channels"]] == ["bigval"]

    @pytest.mark.parametrize(
        "stanza",
        [
            "[notes]\nname=X\ntype=raw\n",  # no id
            "[notes]\nid=1\ntype=raw\n",  # no name
            "[notes]\nid=1\nname=X\n",  # no type
            "[instrument_info]\nid=1\nname=X\ntype=raw\n",  # reserved stanza
            "[cruise info]\nid=1\nname=X\ntype=raw\n",  # reserved, folded name
            "[notes]\nid=\nname=X\ntype=raw\n",  # empty id: setupstr.m never sees it
            "[notes]\nid=1\nname=\ntype=raw\n",  # empty name
            "[notes]\nid_even=1\nid_odd=\nname=X\ntype=raw\n",  # half an id pair
        ],
    )
    def test_incomplete_or_reserved_stanza_is_not_a_channel(self, stanza):
        assert parse_config(_MATRIX + "\n" + stanza)["channels"] == []

    def test_root_level_keys_never_make_a_channel(self):
        assert parse_config("id=1\nname=X\ntype=raw\n" + _MATRIX)["channels"] == []

    def test_bare_channel_stanza_unchanged(self):
        # A [channel] stanza is a channel even without name/type, as before;
        # PFile reports the gap where it converts channels.
        assert parse_config(_MATRIX + "\n[channel]\nid=1\n")["channels"] == [{"id": "1"}]

    def test_duplicate_named_stanzas_stay_separate(self):
        text = _MATRIX + "\n[ch]\nid=1\nname=A\ntype=raw\n\n[ch]\nid=2\nname=B\ntype=raw\n"
        assert [c["name"] for c in parse_config(text)["channels"]] == ["A", "B"]

    def test_piezo_rewrite_applies_to_named_stanza(self):
        text = _MATRIX + "\n[pitch]\nid=1\ntype=accel\nname=Ax\ncoef0=0\ncoef1=1\n"
        assert parse_config(text)["channels"][0]["type"] == "piezo"

    def test_is_channel_section_rule(self):
        assert is_channel_section("channel", {})
        assert is_channel_section("shear1", {"id": "8", "name": "sh1", "type": "shear"})
        assert not is_channel_section("matrix", {"id": "8", "name": "sh1", "type": "shear"})
        assert not is_channel_section("root", {"id": "8", "name": "sh1", "type": "shear"})


class TestPFile:
    def _write(self, path, text):
        rec = np.arange((256 - 128) // 2, dtype=np.int16)
        _make_minimal_p_file(
            path, config_text=text, fast_cols=4, slow_cols=0, n_rows=4, n_records=2, record_data=rec
        )

    def test_named_config_converts_like_modern(self, tmp_path):
        named, modern = tmp_path / "named.p", tmp_path / "modern.p"
        self._write(named, _NAMED)
        self._write(modern, _MODERN)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no "no usable [channel] section" warning
            pf_named = PFile(named)
        pf_modern = PFile(modern)
        assert sorted(pf_named.channels) == sorted(pf_modern.channels) == ["Ax", "Ay", "Gnd", "sh1"]
        for name in pf_modern.channels:
            np.testing.assert_array_equal(pf_named.channels[name], pf_modern.channels[name])

    def test_named_id_even_odd_converts_like_explicit_id(self, tmp_path):
        matrix = "\n[matrix]\nrow1 = 1 2\nrow2 = 1 2\nrow3 = 1 2\nrow4 = 1 2\n"
        rec = np.arange((256 - 128) // 2, dtype=np.int16)
        rec[1::2] = -1  # high (odd) words, so a swapped even/odd join would differ
        explicit, named = tmp_path / "explicit.p", tmp_path / "named.p"
        for path, stanza in [
            (explicit, "\n[channel]\nid = 1, 2\nname = bigval\ntype = raw\n"),
            (named, "\n[big]\nid_even = 1\nid_odd = 2\nname = bigval\ntype = raw\n"),
        ]:
            _make_minimal_p_file(
                path,
                config_text=matrix + stanza,
                fast_cols=2,
                slow_cols=0,
                n_rows=4,
                n_records=1,
                record_data=rec,
            )
        np.testing.assert_array_equal(
            PFile(named).channels["bigval"], PFile(explicit).channels["bigval"]
        )


class TestConfigPatch:
    def test_patch_named_stanza_by_channel_name(self):
        spec = cp.EditSpec(note="sheet value", author="t", channels={"sh1": {"sens": "0.0812"}})
        text, changes = cp.edit_config_text(_NAMED, spec)
        assert len(changes) == 1
        assert parse_config(text)["channels"][3]["sens"] == "0.0812"
        # the edit landed inside [shear1], not appended elsewhere
        assert parse_config(text)["shear1"]["sens"] == "0.0812"

    def test_unknown_channel_still_rejected(self):
        spec = cp.EditSpec(note="n", author="t", channels={"nope": {"sens": "0.08"}})
        with pytest.raises(ValueError):
            cp.edit_config_text(_NAMED, spec)
