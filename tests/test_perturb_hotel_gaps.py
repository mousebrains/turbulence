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

These tests pin both the new gating and --- just as important --- that the
defaults still reproduce the old numbers exactly.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from odas_tpw.perturb.hotel import HotelData, _interp_one, interpolate_hotel


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


def test_default_still_fabricates_exactly_as_before():
    """The knobs must not change what an existing config already produced."""
    t, v = _sparse()
    mid = np.linspace(1500.0, 2100.0, 7)
    out = _interp_one(t, v, mid, "linear")
    assert np.all(np.isfinite(out))
    # The historical values, to four places.
    np.testing.assert_allclose(
        out, [19.1638, 19.2036, 19.2434, 19.2833, 19.3231, 19.3629, 19.4027],
        atol=1e-4,
    )
    for kind in ("pchip", "linear"):
        assert np.all(np.isfinite(_interp_one(t, v, np.linspace(-100, 7500, 200), kind)))


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


def test_extrapolate_false_nans_outside_coverage():
    t, v = _sparse()
    past = np.array([20000.0, 20600.0])
    assert np.all(_interp_one(t, v, past, "linear") == v[-1])
    assert np.all(np.isnan(_interp_one(t, v, past, "linear", extrapolate=False)))


def test_builder_side_nan_marking_now_survives_the_merge():
    t, v = _sparse()
    t = np.concatenate([t, [1800.0]])
    v = np.concatenate([v, [np.nan]])
    o = np.argsort(t)
    t, v = t[o], v[o]
    mid = np.linspace(1500.0, 2100.0, 7)
    assert np.all(np.isfinite(_interp_one(t, v, mid, "linear")))       # old
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
    assert stats["n_gap"] > 0
    assert stats["median_dt"] == pytest.approx(1.0)
    assert stats["widest_gap"] > 3000.0


def test_dense_channel_reports_nothing_to_warn_about():
    """Ordinary sampling jitter must not trip the warning."""
    t = np.arange(0.0, 600.0)
    v = np.sin(t / 60.0)
    stats: dict = {}
    _interp_one(t, v, np.linspace(0.0, 599.0, 500), "linear", stats=stats)
    assert stats["n_gap"] == 0
    assert stats["n_outside"] == 0


def test_interpolate_hotel_warns_about_fabricated_samples():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"sci_water_temp": v}, time_is_relative=True)
    pf = _PF(np.linspace(0.0, 7260.0, 500))
    with pytest.warns(UserWarning, match="interpolated across gaps"):
        interpolate_hotel(hd, pf, {})


def test_interpolate_hotel_warns_about_edge_holding():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v}, time_is_relative=True)
    pf = _PF(np.linspace(7300.0, 9000.0, 50))
    with pytest.warns(UserWarning, match="edge-held outside coverage"):
        interpolate_hotel(hd, pf, {})


def test_per_channel_max_gap_overrides_the_global():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v, "b": v}, time_is_relative=True)
    pf = _PF(np.array([1800.0, 1801.0]))
    out = interpolate_hotel(hd, pf, {
        "max_gap": 30.0,
        "channels": {"a": {}, "b": {"max_gap": None}},
    })
    assert np.isnan(out["a"]).all()      # takes the global gate
    assert np.isfinite(out["b"]).all()   # opts out of it


def test_per_channel_extrapolate_overrides_the_global():
    t, v = _sparse()
    hd = HotelData(time=t, channels={"a": v, "b": v}, time_is_relative=True)
    pf = _PF(np.array([20000.0]))
    out = interpolate_hotel(hd, pf, {
        "extrapolate": False,
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
