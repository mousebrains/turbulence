# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Gap control in the hotel merge.

The merge used to interpolate across arbitrary gaps and edge-hold outside
coverage with no way to say otherwise.  On an instrument whose CTD ran on only
some profiles that produced a smooth **fabricated ramp** between real samples
hours apart, and every consumer --- ``ct``, ``ctd``, ``stratification``,
``salinity: "measured"``, ``epsilon.T_source`` --- read it as data.

Worse, a builder that NaN-marked a dropout (``dinkum-hotel``'s
``projection.max_gap``) had that undone here: the loader dropped the NaN and
interpolated across it anyway, so builder-side gap control produced
byte-identical output.

``hotel.max_gap`` is now **required**: there is no safe default, because the
right limit is the sensor's own rate.  ``hotel.extrapolate`` now defaults to
False.  The old behaviour is still reachable --- deliberately, by typing
``max_gap: "unlimited"`` --- so a historical result can be regenerated.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from odas_tpw.perturb.hotel import (
    HotelData,
    _bridged_gap,
    _interp_one,
    _warn_if_fabricated,
    interpolate_hotel,
)


class _PF:
    """Minimal PFile-like object with a caller-chosen time grid."""

    def __init__(self, t, start_epoch=0.0):
        self.fs_fast = 512
        self.fs_slow = 64
        self.t_fast = np.asarray(t, dtype=float)
        self.t_slow = np.asarray(t, dtype=float)
        self.start_time = datetime.fromtimestamp(start_epoch, tz=UTC)
        self.channels = {}


def _sparse():
    """Three 60 s bursts of 1 Hz samples, an hour apart — CT on every 3rd yo."""
    t, v = [], []
    for k, base in enumerate((0.0, 3600.0, 7200.0)):
        t.append(base + np.arange(60.0))
        v.append(18.0 + 2.0 * k + 0.01 * np.arange(60))
    return np.concatenate(t), np.concatenate(v)


def test_legacy_semantics_are_still_reachable():
    """`max_gap: "unlimited"` + `extrapolate: True` reproduces the old numbers.

    Kept so a historical result can still be regenerated deliberately.
    """
    t, v = _sparse()
    mid = np.linspace(1500.0, 2100.0, 7)
    out = _interp_one(t, v, mid, "linear", max_gap=None, extrapolate=True)
    np.testing.assert_allclose(
        out, [19.1638, 19.2036, 19.2434, 19.2833, 19.3231, 19.3629, 19.4027],
        atol=1e-4,
    )
    for kind in ("pchip", "linear"):
        assert np.all(np.isfinite(_interp_one(
            t, v, np.linspace(-100, 7500, 200), kind,
            max_gap=None, extrapolate=True)))


def test_max_gap_nans_a_fabricated_span():
    t, v = _sparse()
    mid = np.linspace(1500.0, 2100.0, 7)
    assert np.all(np.isnan(_interp_one(t, v, mid, "linear", max_gap=30.0)))


def test_max_gap_keeps_genuinely_sampled_points():
    t, v = _sparse()
    out = _interp_one(t, v, np.array([10.0, 20.5, 3610.0]), "linear",
                      max_gap=30.0, extrapolate=False)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(18.10)


def test_extrapolation_is_off_by_default():
    t, v = _sparse()
    past = np.array([20000.0, 20600.0])
    assert np.all(np.isnan(_interp_one(t, v, past, "linear")))
    assert np.all(_interp_one(t, v, past, "linear", extrapolate=True) == v[-1])


def test_builder_side_nan_marking_now_survives_the_merge():
    t, v = _sparse()
    t = np.concatenate([t, [1800.0]])
    v = np.concatenate([v, [np.nan]])
    o = np.argsort(t)
    t, v = t[o], v[o]
    mid = np.linspace(1500.0, 2100.0, 7)
    assert np.all(np.isfinite(_interp_one(t, v, mid, "linear")))       # ungated
    assert np.all(np.isnan(_interp_one(t, v, mid, "linear", max_gap=30.0)))


def test_pchip_honours_the_gates_too():
    t, v = _sparse()
    mid = np.linspace(1500.0, 2100.0, 5)
    assert np.all(np.isnan(_interp_one(t, v, mid, "pchip", max_gap=30.0)))
    past = np.array([20000.0])
    assert np.all(np.isnan(_interp_one(t, v, past, "pchip", extrapolate=False)))


def test_gap_stats_are_reported():
    t, v = _sparse()
    stats: dict = {}
    _interp_one(t, v, np.linspace(-50.0, 7400.0, 100), "linear", stats=stats)
    assert stats["n_outside"] > 0
    assert stats["n_notable"] > 0
    assert stats["n_rejected"] == 0        # ungated: nothing was thrown away
    assert stats["median_dt"] == pytest.approx(1.0)
    assert stats["widest_gap"] > 3000.0


def test_dense_channel_reports_nothing_to_warn_about():
    """Ordinary sampling jitter must not trip the warning."""
    t = np.arange(0.0, 600.0)
    v = np.sin(t / 60.0)
    stats: dict = {}
    _interp_one(t, v, np.linspace(0.0, 599.0, 500), "linear", stats=stats)
    assert stats["n_notable"] == 0
    assert stats["n_rejected"] == 0
    assert stats["n_outside"] == 0


# --- review findings on this PR -------------------------------------------
def test_a_measured_sample_is_never_rejected_as_a_gap():
    """A target landing exactly ON a source sample is data, not interpolation.

    searchsorted(side="left") returns the index of the match, so without an
    exact-match check the first real sample after a dropout inherits the
    dropout's width and max_gap throws away an observation.
    """
    t = np.array([0.0, 1.0, 100.0])
    v = np.array([10.0, 11.0, 20.0])
    tgt = np.array([1.0, 100.0])
    np.testing.assert_allclose(_bridged_gap(t, tgt), [0.0, 0.0])
    out = _interp_one(t, v, tgt, "linear", max_gap=10.0)
    np.testing.assert_allclose(out, [11.0, 20.0])


def test_shared_clock_is_not_gated_at_all():
    """If source and target share a clock, every target is a measured sample."""
    t = np.concatenate([np.arange(0.0, 10.0), np.arange(3600.0, 3610.0)])
    v = np.arange(t.size, dtype=float)
    out = _interp_one(t, v, t.copy(), "linear", max_gap=30.0)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, v)


def test_warning_does_not_claim_rejection_that_did_not_happen():
    """max_gap above the notable threshold keeps the data; the text must agree."""
    t = np.array([0.0, 1.0, 2.0, 22.0, 23.0, 24.0])
    v = np.arange(6.0)
    stats: dict = {}
    out = _interp_one(t, v, np.array([10.0]), "linear", max_gap=100.0, stats=stats)
    assert np.all(np.isfinite(out))          # nothing was rejected
    assert stats["n_rejected"] == 0
    assert stats["n_notable"] == 1
    with pytest.warns(UserWarning, match="interpolated across gaps"):
        _warn_if_fabricated("x", stats, 100.0, False)


def test_warning_does_claim_rejection_when_it_happens():
    t = np.array([0.0, 1.0, 2.0, 22.0, 23.0, 24.0])
    v = np.arange(6.0)
    stats: dict = {}
    out = _interp_one(t, v, np.array([10.0]), "linear", max_gap=5.0, stats=stats)
    assert np.all(np.isnan(out))
    assert stats["n_rejected"] == 1
    with pytest.warns(UserWarning, match=r"NaN-ed for falling in a gap"):
        _warn_if_fabricated("x", stats, 5.0, False)


def test_interpolate_hotel_warns_about_fabricated_samples():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"sci_water_temp": v}, time_is_relative=True)
    pf = _PF(np.linspace(0.0, 7260.0, 500))
    with pytest.warns(UserWarning, match="interpolated across gaps"):
        interpolate_hotel(hd, pf, {"max_gap": "unlimited"})


def test_interpolate_hotel_warns_about_edge_holding():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v}, time_is_relative=True)
    pf = _PF(np.linspace(7300.0, 9000.0, 50))
    with pytest.warns(UserWarning, match="edge-held outside coverage"):
        interpolate_hotel(hd, pf, {"max_gap": "unlimited", "extrapolate": True})


def test_per_channel_max_gap_overrides_the_global():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v, "b": v}, time_is_relative=True)
    pf = _PF(np.array([1800.0, 1801.0]))
    out = interpolate_hotel(hd, pf, {
        "max_gap": 30.0,
        "channels": {"a": {}, "b": {"max_gap": "unlimited"}},
    })
    assert np.isnan(out["a"]).all()      # takes the global gate
    assert np.isfinite(out["b"]).all()   # opts out of it


def test_per_channel_extrapolate_overrides_the_global():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v, "b": v}, time_is_relative=True)
    pf = _PF(np.array([20000.0]))
    out = interpolate_hotel(hd, pf, {
        "max_gap": 30.0,
        "channels": {"a": {}, "b": {"extrapolate": True}},
    })
    assert np.isnan(out["a"]).all()
    assert np.isfinite(out["b"]).all()


def test_max_gap_and_extrapolate_are_accepted_channel_options():
    """They must survive schema validation rather than being 'unknown options'."""
    from odas_tpw.perturb.hotel import _normalize_channels_cfg

    _active, opts = _normalize_channels_cfg(
        {"x": {"max_gap": 30.0, "extrapolate": False}}
    )
    assert opts["x"]["max_gap"] == 30.0
    assert opts["x"]["extrapolate"] is False


def test_max_gap_is_required():
    """Omitting it is an error, not a default -- there is no safe default."""
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v}, time_is_relative=True)
    pf = _PF(np.array([10.0, 20.0]))
    with pytest.raises(ValueError, match=r"hotel\.max_gap is required"):
        interpolate_hotel(hd, pf, {})


def test_max_gap_rejects_nonsense():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v}, time_is_relative=True)
    pf = _PF(np.array([10.0]))
    with pytest.raises(ValueError, match="expected a number of seconds"):
        interpolate_hotel(hd, pf, {"max_gap": "sometimes"})
    with pytest.raises(ValueError, match="positive number of seconds"):
        interpolate_hotel(hd, pf, {"max_gap": -5})


def test_unlimited_is_case_insensitive_and_opts_out():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v}, time_is_relative=True)
    pf = _PF(np.array([1800.0]))
    out = interpolate_hotel(hd, pf, {"max_gap": "UNLIMITED"})
    assert np.isfinite(out["a"]).all()


def test_per_channel_max_gap_is_validated_too():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v}, time_is_relative=True)
    pf = _PF(np.array([10.0]))
    with pytest.raises(ValueError, match=r"channels\['a'\].max_gap"):
        interpolate_hotel(hd, pf, {"max_gap": 30.0,
                                   "channels": {"a": {"max_gap": "no"}}})


# ---------------------------------------------------------------------------
# Round-2 review items: sorting, inherit-on-null, measured tolerance,
# fail-fast validation, and the gate travelling downstream.
# ---------------------------------------------------------------------------


def test_unsorted_source_times_are_sorted_before_use():
    """An unsorted CSV/NetCDF must not turn into all-NaN output.

    interp1d sorts internally, but the range/gap bookkeeping reads
    hotel_t[0]/hotel_t[-1] directly, so unsorted times used to reject
    everything as "outside coverage"; pchip raised outright.
    """
    hotel_t = np.array([15.0, 0.0, 5.0, 10.0])
    data = 2.0 + hotel_t  # linear in time regardless of storage order
    targets = np.array([0.0, 5.0, 10.0])
    for kind in ("linear", "nearest", "pchip"):
        out = _interp_one(hotel_t, data, targets, kind, max_gap=100.0)
        np.testing.assert_allclose(out, [2.0, 7.0, 12.0])


def test_per_channel_null_inherits_the_global():
    """`max_gap: null` / `extrapolate: null` on a channel mean "inherit"."""
    from odas_tpw.perturb.hotel import resolve_gap_settings

    settings = resolve_gap_settings({
        "max_gap": 30.0,
        "extrapolate": True,
        "channels": {"a": {"max_gap": None, "extrapolate": None}},
    })
    assert settings["a"] == (30.0, True)
    assert settings[None] == (30.0, True)


def test_per_channel_null_does_not_raise_when_global_is_set():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v}, time_is_relative=True)
    pf = _PF(np.array([1800.0, 20000.0]))
    out = interpolate_hotel(hd, pf, {
        "max_gap": 30.0,
        "extrapolate": True,
        "channels": {"a": {"max_gap": None, "extrapolate": None}},
    })
    assert np.isnan(out["a"][0])      # inherited the 30 s gate
    assert np.isfinite(out["a"][1])   # inherited extrapolate: true


def test_measured_match_tolerates_float_slop():
    """A target within ~1e-6 of a source sample is that sample, not the gap.

    Hotel epochs shifted by the file start have ~1e-7 s resolution, so exact
    equality was luck: a target 1e-9 s before the first sample after a 98 s
    dropout was NaN-ed while 1e-9 after was kept.
    """
    hotel_t = np.array([0.0, 1.0, 99.0, 100.0])
    data = hotel_t.copy()
    for slop in (-1e-9, 0.0, 1e-9):
        out = _interp_one(hotel_t, data, np.array([99.0 + slop]), "linear",
                          max_gap=10.0)
        assert np.isfinite(out[0]), f"slop={slop}"
    # A target genuinely inside the dropout is still rejected.
    out = _interp_one(hotel_t, data, np.array([50.0]), "linear", max_gap=10.0)
    assert np.isnan(out[0])


def test_bridged_gap_tolerance_direct():
    hotel_t = np.array([0.0, 1.0, 99.0, 100.0])
    gap = _bridged_gap(hotel_t, np.array([99.0 - 1e-9, 99.0 + 1e-9, 50.0]))
    assert gap[0] == 0.0 and gap[1] == 0.0
    assert gap[2] == 98.0


def test_validate_config_fails_fast_on_missing_max_gap(tmp_path):
    """A missing max_gap errors once at config-load time, not per file."""
    import yaml

    from odas_tpw.perturb.config import load_config

    cfg = tmp_path / "perturb.yaml"
    cfg.write_text(yaml.safe_dump({
        "files": {"p_file_root": str(tmp_path), "output_root": str(tmp_path)},
        "hotel": {"enable": True, "file": "hotel.csv"},
    }))
    with pytest.raises(ValueError, match=r"hotel\.max_gap is required"):
        load_config(str(cfg))


def test_validate_config_checks_per_channel_overrides_too(tmp_path):
    import yaml

    from odas_tpw.perturb.config import load_config

    cfg = tmp_path / "perturb.yaml"
    cfg.write_text(yaml.safe_dump({
        "files": {"p_file_root": str(tmp_path), "output_root": str(tmp_path)},
        "hotel": {"enable": True, "file": "hotel.csv", "max_gap": 30.0,
                  "channels": {"a": {"max_gap": "bogus"}}},
    }))
    with pytest.raises(ValueError, match=r"channels\['a'\]\.max_gap"):
        load_config(str(cfg))


def test_merge_records_max_gap_on_channel_info():
    """The resolved gate travels with the channel for downstream refills."""
    from odas_tpw.perturb.hotel import merge_hotel_into_pfile

    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v, "b": v}, time_is_relative=True)
    pf = _PF(np.linspace(0.0, 7260.0, 500))
    pf.channel_info = {}
    pf._fast_channels = set()
    merge_hotel_into_pfile(hd, pf, {
        "max_gap": 30.0,
        "channels": {"a": {}, "b": {"max_gap": "unlimited"}},
    })
    assert pf.channel_info["a"]["hotel_max_gap"] == 30.0
    assert "hotel_max_gap" not in pf.channel_info["b"]  # unlimited: no gate
