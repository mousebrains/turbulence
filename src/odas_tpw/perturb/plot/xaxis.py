# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""X-axis coordinate methods for perturb section plots.

A pure-geometry kernel: map per-point ``(lat, lon)`` and ``time`` to a 1-D
x-coordinate under a chosen method.  Dimension-agnostic by design — the same
functions serve the per-sample CTD trajectory (``section`` subcommand) and,
later, per-profile section grids (one position per cast).

No NetCDF I/O, no matplotlib, no gridding here.  Non-finite positions
propagate to non-finite x (callers drop those samples before binning); we
never substitute a fill value, so a missing GPS fix can't masquerade as a
real location.

References for the distance maths:
  - Haversine great-circle distance (radial distance from a fixed point).
  - Local equirectangular projection for along-line projection: accurate to
    well under 1% over the few-hundred-km transects this is used for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Mean Earth radius (spherical approximation; the sub-0.3% flattening error is
# negligible against the sampling/binning scales of a microstructure section).
_EARTH_RADIUS_M = 6_371_000.0

# Distance unit -> metres-per-unit scale factor.  1 nm = 1852 m exactly.
_UNIT_PER_M: dict[str, float] = {"m": 1.0, "km": 1.0e-3, "nm": 1.0 / 1852.0}

METHODS: tuple[str, ...] = (
    "time",
    "latitude",
    "longitude",
    "distance_from_point",
    "along_line",
    "signed_distance",
)


@dataclass
class XAxis:
    """Result of :func:`compute`.

    ``x`` is a numeric 1-D coordinate (epoch seconds for ``time``; degrees for
    latitude/longitude; distance in the requested units otherwise).  ``kind``
    is ``"time"`` or ``"spatial"`` so the renderer can pick a date formatter.
    ``cross`` (along_line only) is each sample's cross-track distance in the
    requested units, else ``None``.
    """

    x: np.ndarray
    label: str
    kind: str
    cross: np.ndarray | None = None


def _unit_scale(units: str) -> float:
    try:
        return _UNIT_PER_M[units]
    except KeyError:
        raise ValueError(f"unknown distance units {units!r}; use 'm', 'km', or 'nm'") from None


def _wrap_deg(d: np.ndarray | float) -> np.ndarray:
    """Wrap a longitude difference into [-180, 180) so it is dateline-safe."""
    return (np.asarray(d, dtype=float) + 180.0) % 360.0 - 180.0


def _circmean_deg(lon: np.ndarray) -> float:
    """Circular mean of longitudes [deg], dateline-safe.

    A plain ``nanmean`` of raw degrees averages the *wrapped* values, so a
    transect straddling +/-180 lands on the antipode (e.g. [179, -179] -> 0,
    not 180), which then distorts the local projection by ~180x. Averaging the
    unit vectors and taking the angle keeps the reference on the correct side.
    """
    rad = np.radians(np.asarray(lon, dtype=float))
    finite = np.isfinite(rad)
    if not np.any(finite):
        return float("nan")
    return float(
        np.degrees(
            np.arctan2(np.nanmean(np.sin(rad[finite])), np.nanmean(np.cos(rad[finite])))
        )
    )


def to_epoch_seconds(time: np.ndarray) -> np.ndarray:
    """Return *time* as float epoch seconds.

    Accepts datetime64 (decoded CF time) or an already-numeric epoch-seconds
    array.  datetime64 is assumed UTC, matching the CTD product's
    ``seconds since 1970-01-01`` encoding.
    """
    t = np.asarray(time)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").astype("int64") / 1.0e9
    return t.astype(float)


def haversine_m(
    lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """Great-circle distance [m] from the fixed point (lat0, lon0) to each point.

    NaN positions propagate to NaN distance.  Dateline-safe via longitude
    wrapping.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    phi0 = np.radians(lat0)
    phi = np.radians(lat)
    dphi = np.radians(lat - lat0)
    dlmb = np.radians(_wrap_deg(lon - lon0))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi0) * np.cos(phi) * np.sin(dlmb / 2.0) ** 2
    dist = 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.asarray(dist, dtype=float)


def distance_from_point(
    lat: np.ndarray, lon: np.ndarray, point: tuple[float, float], units: str = "km"
) -> np.ndarray:
    """Radial great-circle distance from a fixed (lat, lon) reference point.

    *point* is ``(lat, lon)`` in degrees.  This is N independent distances to
    one fixed point — deliberately a haversine rather than ``gsw.distance``
    (which returns N-1 distances between *consecutive* points and takes its
    arguments lon-first).
    """
    lat0, lon0 = float(point[0]), float(point[1])
    return haversine_m(lat0, lon0, lat, lon) * _unit_scale(units)


def _to_local_xy(
    lat: np.ndarray, lon: np.ndarray, lat_ref: float, lon_ref: float
) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular (east, north) metres about (lat_ref, lon_ref)."""
    x = np.radians(_wrap_deg(np.asarray(lon, float) - lon_ref)) * np.cos(
        np.radians(lat_ref)
    ) * _EARTH_RADIUS_M
    y = np.radians(np.asarray(lat, float) - lat_ref) * _EARTH_RADIUS_M
    return x, y


def project_along_line(
    lat: np.ndarray,
    lon: np.ndarray,
    waypoints: np.ndarray,
    units: str = "km",
) -> tuple[np.ndarray, np.ndarray]:
    """Project each (lat, lon) onto a waypoint polyline.

    Returns ``(along, cross)``: cumulative along-line distance to the foot of
    the perpendicular on the nearest segment, and the cross-track distance to
    that foot, both in *units*.  The foot is clamped to each segment, so
    samples beyond the polyline ends fold onto the terminal waypoint.

    *waypoints* is an ``(M, 2)`` array of ``[lat, lon]`` rows (M >= 2).  The
    projection uses a local equirectangular plane centred on the data
    centroid — exact enough for the few-hundred-km transects this serves, but
    an approximation, not a geodesic.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    wp = np.asarray(waypoints, dtype=float)
    if wp.ndim != 2 or wp.shape[0] < 2 or wp.shape[1] != 2:
        raise ValueError("along_line needs >= 2 waypoints as [[lat, lon], ...]")

    lat_ref = float(np.nanmean(lat)) if np.any(np.isfinite(lat)) else float(wp[0, 0])
    # Circular mean so a dateline-crossing transect's reference stays on the
    # correct side of +/-180 (bug_001).
    lon_ref = _circmean_deg(lon) if np.any(np.isfinite(lon)) else float(wp[0, 1])

    px, py = _to_local_xy(lat, lon, lat_ref, lon_ref)
    wx, wy = _to_local_xy(wp[:, 0], wp[:, 1], lat_ref, lon_ref)

    seg_dx = np.diff(wx)
    seg_dy = np.diff(wy)
    seg_len2 = seg_dx**2 + seg_dy**2
    seg_len = np.sqrt(seg_len2)
    cumlen = np.concatenate(([0.0], np.cumsum(seg_len)))

    n = px.size
    finite = np.isfinite(px) & np.isfinite(py)
    best_along = np.full(n, np.nan)
    best_cross = np.full(n, np.inf)
    for s in range(seg_dx.size):
        if seg_len2[s] == 0.0:
            continue  # zero-length segment (duplicate waypoint)
        t = ((px - wx[s]) * seg_dx[s] + (py - wy[s]) * seg_dy[s]) / seg_len2[s]
        t = np.clip(t, 0.0, 1.0)
        fx = wx[s] + t * seg_dx[s]
        fy = wy[s] + t * seg_dy[s]
        cross = np.hypot(px - fx, py - fy)
        along = cumlen[s] + t * seg_len[s]
        upd = finite & (cross < best_cross)
        best_along = np.where(upd, along, best_along)
        best_cross = np.where(upd, cross, best_cross)

    best_cross = np.where(np.isfinite(best_along), best_cross, np.nan)
    scale = _unit_scale(units)
    return best_along * scale, best_cross * scale


@dataclass
class SignedAxis:
    """Result of :func:`signed_distance_axis`.

    ``x`` is the signed along-axis distance from the midpoint.  ``mid_lat`` /
    ``mid_lon`` are the centroid (the midpoint).  ``bearing_deg`` is the
    orientation of *increasing distance* clockwise from true north;
    ``bearing_std_deg`` is the circular standard deviation of the points'
    orientation about the best-fit line.  Orientation fields are NaN when the
    track has no spatial spread.
    """

    x: np.ndarray
    mid_lat: float
    mid_lon: float
    bearing_deg: float
    bearing_std_deg: float


def signed_distance_axis(
    lat: np.ndarray, lon: np.ndarray, time: np.ndarray, units: str = "km"
) -> SignedAxis:
    """Project (lat, lon) onto the best-fit principal axis of the points.

    Returns the signed along-axis distance from the centroid (earliest sample
    negative, latest positive) plus the midpoint and the axis orientation
    relative to true north with a circular standard deviation.

    Orientation and spread come from the orientation tensor (a total-least-
    squares line fit through the centroid).  With centroid-relative
    displacements ``z = east + i*north``, ``S = sum(z**2)`` has argument
    ``2*theta`` (theta = major-axis angle) and modulus-ratio
    ``R = |S| / sum|z|**2`` is the doubled-angle resultant length; the circular
    standard deviation of the (undirected) orientation is ``sqrt(-2 ln R) / 2``.

    Non-finite positions propagate to NaN x.  A single station (no spatial
    spread) yields all-zero distances and NaN orientation.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    nan = float("nan")
    if not np.any(finite):
        return SignedAxis(np.full(lat.shape, np.nan), nan, nan, nan, nan)

    mid_lat = float(np.nanmean(lat))
    # Circular mean so a dateline-crossing transect's reference (and the
    # midpoint label) stays on the correct side of +/-180 (bug_001).
    mid_lon = _circmean_deg(lon)
    east, north = _to_local_xy(lat, lon, mid_lat, mid_lon)
    de = east - float(np.mean(east[finite]))   # east displacement from centroid
    dn = north - float(np.mean(north[finite]))  # north displacement
    ef, nf = de[finite], dn[finite]

    sumsq = float(np.sum(ef**2 + nf**2))
    if sumsq <= 0.0:  # single station / no spatial spread
        return SignedAxis(np.where(finite, 0.0, np.nan), mid_lat, mid_lon, nan, nan)

    # Orientation tensor via the doubled-angle complex sum.
    s = complex(np.sum((ef + 1j * nf) ** 2))
    phi = float(np.angle(s)) / 2.0               # major-axis angle from east
    axis = np.array([np.cos(phi), np.sin(phi)])  # (east, north) unit vector

    proj = de * axis[0] + dn * axis[1]
    # Orient earliest -> negative, latest -> positive.
    tf = to_epoch_seconds(time)[finite]
    pf = proj[finite]
    if np.std(pf) > 0 and np.std(tf) > 0 and float(np.cov(pf, tf)[0, 1]) < 0:
        proj = -proj
        axis = -axis

    bearing = float(np.degrees(np.arctan2(axis[0], axis[1])) % 360.0)  # CW from N
    r = min(max(abs(s) / sumsq, 1e-12), 1.0)
    std_deg = float(np.degrees(np.sqrt(-2.0 * np.log(r)) / 2.0))
    return SignedAxis(
        np.asarray(proj * _unit_scale(units), dtype=float),
        mid_lat, mid_lon, bearing, std_deg,
    )


def signed_distance_from_midpoint(
    lat: np.ndarray, lon: np.ndarray, time: np.ndarray, units: str = "km"
) -> np.ndarray:
    """Signed distance from the midpoint along the points' principal axis.

    Thin wrapper over :func:`signed_distance_axis` returning only the
    coordinate (earliest negative, latest positive; non-finite -> NaN).
    """
    return signed_distance_axis(lat, lon, time, units).x


def _fmt_lat_lon(lat: float, lon: float) -> str:
    """Latitude/longitude to 4 significant digits with hemisphere suffixes."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4g}°{ns}, {abs(lon):.4g}°{ew}"


def compute(
    method: str,
    lat: np.ndarray,
    lon: np.ndarray,
    time: np.ndarray,
    params: dict | None = None,
) -> XAxis:
    """Dispatch to the requested x-axis method.

    *params* carries method-specific options: ``units`` (distance methods),
    ``point`` ([lat, lon], distance_from_point), ``waypoints`` ([[lat,lon]..],
    along_line).  Raises ``ValueError`` on an unknown method or missing param.
    """
    params = params or {}
    units = params.get("units", "km")

    if method == "time":
        return XAxis(x=to_epoch_seconds(time), label="Time (UTC)", kind="time")
    if method == "latitude":
        return XAxis(x=np.asarray(lat, float), label=r"Latitude ($^{\circ}$N)", kind="spatial")
    if method == "longitude":
        return XAxis(x=np.asarray(lon, float), label=r"Longitude ($^{\circ}$E)", kind="spatial")
    if method == "distance_from_point":
        point = params.get("point")
        if point is None or len(point) != 2:
            raise ValueError("distance_from_point needs params['point'] = [lat, lon]")
        x = distance_from_point(lat, lon, (float(point[0]), float(point[1])), units)
        label = f"Distance from ({float(point[0]):.4f}, {float(point[1]):.4f}) [{units}]"
        return XAxis(x=x, label=label, kind="spatial")
    if method == "along_line":
        waypoints = params.get("waypoints")
        if waypoints is None:
            raise ValueError("along_line needs params['waypoints'] = [[lat, lon], ...]")
        along, cross = project_along_line(lat, lon, np.asarray(waypoints, float), units)
        return XAxis(
            x=along, label=f"Along-track distance [{units}]", kind="spatial", cross=cross
        )
    if method == "signed_distance":
        r = signed_distance_axis(lat, lon, time, units)
        if np.isfinite(r.mid_lat) and np.isfinite(r.bearing_deg):
            label = (
                f"Signed distance from {_fmt_lat_lon(r.mid_lat, r.mid_lon)} [{units}], "
                f"orientation {r.bearing_deg:.1f}°±{r.bearing_std_deg:.1f}° T"
            )
        elif np.isfinite(r.mid_lat):
            label = f"Signed distance from {_fmt_lat_lon(r.mid_lat, r.mid_lon)} [{units}]"
        else:
            label = f"Signed distance from midpoint [{units}]"
        return XAxis(x=r.x, label=label, kind="spatial")
    raise ValueError(f"unknown x-axis method {method!r}; choose from {METHODS}")
