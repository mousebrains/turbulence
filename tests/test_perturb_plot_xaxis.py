# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Unit tests for the section x-axis geometry kernel (plot/xaxis.py)."""

from __future__ import annotations

import numpy as np
import pytest

from odas_tpw.perturb.plot import xaxis

# One degree of great-circle arc on the mean-radius sphere.
_DEG_KM = 6_371_000.0 * np.pi / 180.0 / 1000.0  # ~111.19 km


def test_haversine_one_degree_latitude():
    d = xaxis.haversine_m(0.0, 0.0, np.array([1.0]), np.array([0.0]))
    assert d[0] == pytest.approx(_DEG_KM * 1000.0, rel=1e-6)


def test_haversine_longitude_shrinks_with_latitude():
    """1 deg of longitude at 60N is ~half that at the equator (cos 60 = 0.5)."""
    eq = xaxis.haversine_m(0.0, 0.0, np.array([0.0]), np.array([1.0]))[0]
    hi = xaxis.haversine_m(60.0, 0.0, np.array([60.0]), np.array([1.0]))[0]
    assert hi / eq == pytest.approx(0.5, rel=1e-3)


def test_haversine_dateline_safe():
    """Across +/-179.9 the distance is ~0.2 deg, not ~360 deg."""
    d = xaxis.haversine_m(0.0, 179.9, np.array([0.0]), np.array([-179.9]))[0]
    assert d == pytest.approx(_DEG_KM * 1000.0 * 0.2, rel=1e-3)


def test_distance_from_point_is_radial_not_cumulative():
    """N distances to one fixed point, not N-1 consecutive deltas."""
    lat = np.array([0.0, 1.0, 2.0])
    lon = np.zeros(3)
    d = xaxis.distance_from_point(lat, lon, (0.0, 0.0), units="km")
    assert d == pytest.approx(np.array([0.0, _DEG_KM, 2 * _DEG_KM]), rel=1e-5)


def test_distance_units_scale():
    lat = np.array([1.0])
    lon = np.array([0.0])
    km = xaxis.distance_from_point(lat, lon, (0.0, 0.0), units="km")[0]
    m = xaxis.distance_from_point(lat, lon, (0.0, 0.0), units="m")[0]
    nm = xaxis.distance_from_point(lat, lon, (0.0, 0.0), units="nm")[0]
    assert m == pytest.approx(km * 1000.0, rel=1e-9)
    # 1 deg of latitude is ~60 nautical miles by historical definition.
    assert nm == pytest.approx(60.0, abs=0.1)


def test_distance_propagates_nan_position():
    lat = np.array([0.0, np.nan])
    lon = np.array([0.0, 0.0])
    d = xaxis.distance_from_point(lat, lon, (0.0, 0.0))
    assert np.isfinite(d[0]) and np.isnan(d[1])


def test_along_line_on_a_straight_meridian():
    """Points on a N-S line: along ~ distance from start, cross ~ 0."""
    waypoints = np.array([[0.0, 0.0], [1.0, 0.0]])
    lat = np.array([0.0, 0.5, 1.0])
    lon = np.zeros(3)
    along, cross = xaxis.project_along_line(lat, lon, waypoints, units="km")
    assert along == pytest.approx(np.array([0.0, 0.5 * _DEG_KM, _DEG_KM]), rel=1e-3)
    assert np.allclose(cross, 0.0, atol=1e-6)


def test_along_line_cross_track_for_off_line_point():
    waypoints = np.array([[0.0, 0.0], [1.0, 0.0]])
    # 0.1 deg east of the midpoint of the meridian.
    along, cross = xaxis.project_along_line(
        np.array([0.5]), np.array([0.1]), waypoints, units="km"
    )
    assert along[0] == pytest.approx(0.5 * _DEG_KM, rel=1e-2)
    assert cross[0] == pytest.approx(0.1 * _DEG_KM * np.cos(np.radians(0.5)), rel=1e-2)


def test_along_line_clamps_beyond_ends():
    """A point south of the start clamps to along=0 (no negative overshoot)."""
    waypoints = np.array([[0.0, 0.0], [1.0, 0.0]])
    along, _cross = xaxis.project_along_line(
        np.array([-0.5]), np.array([0.0]), waypoints, units="km"
    )
    assert along[0] == pytest.approx(0.0, abs=1e-6)


def test_along_line_requires_two_waypoints():
    with pytest.raises(ValueError):
        xaxis.project_along_line(np.array([0.0]), np.array([0.0]), np.array([[0.0, 0.0]]))


def test_compute_time_returns_epoch_seconds():
    t = np.array(["2025-01-01T00:00:00", "2025-01-01T00:00:10"], dtype="datetime64[ns]")
    xa = xaxis.compute("time", np.zeros(2), np.zeros(2), t)
    assert xa.kind == "time"
    assert xa.x[1] - xa.x[0] == pytest.approx(10.0)


def test_compute_latitude_longitude_passthrough():
    lat = np.array([1.0, 2.0])
    lon = np.array([3.0, 4.0])
    t = np.zeros(2)
    assert np.allclose(xaxis.compute("latitude", lat, lon, t).x, lat)
    assert np.allclose(xaxis.compute("longitude", lat, lon, t).x, lon)


def test_compute_rejects_unknown_method_and_missing_params():
    t = np.zeros(1)
    with pytest.raises(ValueError):
        xaxis.compute("bogus", np.zeros(1), np.zeros(1), t)
    with pytest.raises(ValueError):
        xaxis.compute("distance_from_point", np.zeros(1), np.zeros(1), t, params={})
    with pytest.raises(ValueError):
        xaxis.compute("along_line", np.zeros(1), np.zeros(1), t, params={})


def test_haversine_matches_gsw():
    """Cross-check the haversine against TEOS-10 gsw.distance (<0.5%)."""
    gsw = pytest.importorskip("gsw")
    lat = np.array([20.0, 21.0])
    lon = np.array([130.0, 132.0])
    # gsw.distance takes lon FIRST and returns the single inter-point distance.
    ref = gsw.distance(lon, lat, p=0)[0]
    ours = xaxis.haversine_m(lat[0], lon[0], lat[1:2], lon[1:2])[0]
    assert ours == pytest.approx(ref, rel=5e-3)


def _hourly(n):
    return np.array(
        [np.datetime64("2025-01-01T00:00:00") + np.timedelta64(i, "h") for i in range(n)]
    )


def test_signed_distance_centered_and_time_signed():
    """N-S transect: centred at 0, earliest negative, increasing with time."""
    lat = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    lon = np.zeros(5)
    xa = xaxis.compute("signed_distance", lat, lon, _hourly(5))
    x = xa.x
    assert xa.kind == "spatial"
    assert x.mean() == pytest.approx(0.0, abs=1e-6)          # midpoint origin
    assert x[0] < 0 < x[-1]                                  # earliest -, latest +
    assert np.all(np.diff(x) > 0)                            # monotone with time
    assert np.diff(x).mean() == pytest.approx(_DEG_KM, rel=1e-2)  # ~1 deg/step


def test_signed_distance_sign_follows_time_not_space():
    """Latitude DEcreasing with time: the earliest (northern) sample is still
    negative — the sign tracks time, not the coordinate."""
    lat = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    lon = np.zeros(5)
    x = xaxis.compute("signed_distance", lat, lon, _hourly(5)).x
    assert x[0] < 0 < x[-1]
    assert np.all(np.diff(x) > 0)


def test_signed_distance_uses_principal_axis():
    """An E-W track (lon varies, lat ~constant): the axis is east-west."""
    lon = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    lat = np.array([0.0, 0.001, 0.0, 0.001, 0.0])  # negligible N-S spread
    x = xaxis.compute("signed_distance", lat, lon, _hourly(5), params={"units": "km"}).x
    assert np.all(np.diff(x) > 0)
    assert np.diff(x).mean() == pytest.approx(_DEG_KM, rel=2e-2)  # ~1 deg lon at eq.


def test_signed_distance_nan_position_propagates():
    lat = np.array([0.0, 1.0, np.nan, 3.0])
    lon = np.zeros(4)
    x = xaxis.compute("signed_distance", lat, lon, _hourly(4)).x
    assert np.isnan(x[2])
    assert np.all(np.isfinite(x[[0, 1, 3]]))


def test_signed_distance_axis_bearings():
    """Bearing of increasing distance is CW from true north; clean lines -> std 0."""
    t = _hourly(9)
    north = xaxis.signed_distance_axis(np.linspace(0, 4, 9), np.zeros(9), t)
    east = xaxis.signed_distance_axis(np.zeros(9), np.linspace(0, 4, 9), t)
    south = xaxis.signed_distance_axis(np.linspace(4, 0, 9), np.zeros(9), t)
    assert north.bearing_deg == pytest.approx(0.0, abs=1.0)
    assert east.bearing_deg == pytest.approx(90.0, abs=1.0)
    assert south.bearing_deg == pytest.approx(180.0, abs=1.0)
    for r in (north, east, south):
        assert r.bearing_std_deg == pytest.approx(0.0, abs=0.5)
    # midpoint = centroid
    assert north.mid_lat == pytest.approx(2.0)
    assert east.mid_lon == pytest.approx(2.0)


def test_signed_distance_std_grows_with_scatter():
    t = _hourly(50)
    lat = np.linspace(0.0, 4.0, 50)
    clean = xaxis.signed_distance_axis(lat, np.zeros(50), t)
    wiggle = 0.3 * np.sin(np.linspace(0.0, 6.0, 50))  # perpendicular meander
    noisy = xaxis.signed_distance_axis(lat, wiggle, t)
    assert noisy.bearing_std_deg > clean.bearing_std_deg + 1.0


def test_signed_distance_single_station_has_nan_orientation():
    r = xaxis.signed_distance_axis(np.full(5, 20.0), np.full(5, 130.0), _hourly(5))
    assert np.all(r.x == 0.0)
    assert np.isnan(r.bearing_deg) and np.isnan(r.bearing_std_deg)


def test_signed_distance_label_has_midpoint_and_orientation():
    lat = np.linspace(0.0, 4.0, 9)
    lon = np.linspace(0.0, 2.0, 9)
    lbl = xaxis.compute("signed_distance", lat, lon, _hourly(9)).label
    assert lbl.startswith("Signed distance from")
    assert "2°N, 1°E" in lbl                 # midpoint, 4 sig figs, hemispheres
    assert "orientation" in lbl and "±" in lbl and "° T" in lbl


# ---------------------------------------------------------------------------
# Dateline safety of the local-projection reference (bug_001)
# ---------------------------------------------------------------------------


def test_circmean_deg_is_dateline_safe():
    """Circular mean of [179, -179] is 180, not the antipodal 0 a raw mean gives."""
    assert xaxis._circmean_deg(np.array([179.0, 179.9, -179.9, -179.0])) == pytest.approx(
        180.0, abs=1e-6
    )
    # Plain nanmean would land near 0 — confirm we are NOT doing that.
    assert abs(float(np.nanmean(np.array([179.0, -179.0])))) < 1e-6


def test_signed_distance_dateline_span_matches_haversine():
    """A transect across +/-180 must not blow the x-axis span up ~180x; the span
    must match the great-circle ground truth and the midpoint stay near 180 (bug_001)."""
    lat = np.full(4, 20.0)
    lon = np.array([179.0, 179.9, -179.9, -179.0])
    ax = xaxis.signed_distance_axis(lat, lon, _hourly(4), units="km")
    span = float(np.nanmax(ax.x) - np.nanmin(ax.x))
    truth = xaxis.haversine_m(20.0, 179.0, 20.0, -179.0) / 1000.0
    assert span == pytest.approx(truth, rel=0.05)  # ~209 km, not ~37,595 km
    assert ax.mid_lon == pytest.approx(180.0, abs=0.1)


def test_along_line_dateline_span_matches_haversine():
    """project_along_line across ±180 keeps along-track within the true arc and
    cross-track near zero, instead of an antipodal reference distortion (bug_001)."""
    lat = np.full(4, 20.0)
    lon = np.array([179.0, 179.9, -179.9, -179.0])
    waypoints = np.array([[20.0, 179.0], [20.0, -179.0]])
    along, cross = xaxis.project_along_line(lat, lon, waypoints, units="km")
    truth = xaxis.haversine_m(20.0, 179.0, 20.0, -179.0) / 1000.0
    assert np.nanmax(along) <= truth * 1.05
    assert np.nanmax(np.abs(cross)) < 1.0
