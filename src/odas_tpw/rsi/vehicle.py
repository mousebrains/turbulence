# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Vehicle attributes for RSI instruments.

Maps vehicle types to their default profile direction and smoothing
time constant (tau), matching ODAS ``default_vehicle_attributes.ini``.
"""

from __future__ import annotations

# (direction, tau) per vehicle — matches ODAS default_vehicle_attributes.ini
VEHICLE_ATTRIBUTES: dict[str, tuple[str, float]] = {
    "vmp": ("down", 1.5),
    "rvmp": ("up", 1.5),
    "xmp": ("down", 1.5),
    "argo_float": ("up", 60.0),
    "slocum_glider": ("glide", 3.0),
    "sea_glider": ("glide", 5.0),
    "sea_explorer": ("glide", 3.0),
    "auv": ("horizontal", 10.0),
    "auv_emc": ("horizontal", 10.0),
    "nemo": ("horizontal", 60.0),
    "micro_squid": ("horizontal", 1.5),
    "stand": ("horizontal", 1.5),
}

DEFAULT_DIRECTION = "down"
DEFAULT_TAU = 1.5

# Default fall-rate floor [dbar/s] for profile detection. A free-falling VMP
# clears 0.3 easily; slow near-horizontal platforms (gliders, AUVs) do not, so
# their vertical rate needs a lower floor or every cast is rejected. ODAS leaves
# W_min to the caller (its default_vehicle_attributes.ini has no W_min); this is
# a local default keyed on the resolved profile direction.
DEFAULT_W_MIN = 0.3
SLOW_W_MIN = 0.05

_EXPLICIT_DIRECTIONS = {"up", "down", "glide", "horizontal"}
_SLOW_DIRECTIONS = {"glide", "horizontal"}


def resolve_direction(direction: str, vehicle: str) -> str:
    """Resolve profile direction from user setting and vehicle type.

    Parameters
    ----------
    direction : str
        User-specified direction.  ``"auto"`` looks up *vehicle* in
        :data:`VEHICLE_ATTRIBUTES`; any explicit value (``"up"``,
        ``"down"``, ``"glide"``, ``"horizontal"``) passes through unchanged.
    vehicle : str
        Vehicle identifier from the instrument config (lower-case).

    Returns
    -------
    str
        Resolved direction.
    """
    if direction.lower() in _EXPLICIT_DIRECTIONS:
        return direction.lower()
    # "auto" or unrecognised → look up vehicle
    attrs = VEHICLE_ATTRIBUTES.get(vehicle.lower())
    if attrs is not None:
        return attrs[0]
    return DEFAULT_DIRECTION


def resolve_w_min(direction: str) -> float:
    """Return the default fall-rate floor [dbar/s] for a resolved *direction*.

    Slow, near-horizontal platforms (``glide``/``horizontal``) use a lower floor
    (:data:`SLOW_W_MIN`) than a free-falling profiler (:data:`DEFAULT_W_MIN`),
    because their vertical rate is far below the VMP-tuned 0.3 dbar/s default and
    would otherwise reject every cast. Callers passing an explicit ``W_min``
    (e.g. ``--W-min``) bypass this. Expects an already-resolved direction, not
    ``"auto"``.
    """
    return SLOW_W_MIN if direction.lower() in _SLOW_DIRECTIONS else DEFAULT_W_MIN


def resolve_tau(vehicle: str) -> float:
    """Return the smoothing time constant for a vehicle.

    Parameters
    ----------
    vehicle : str
        Vehicle identifier (lower-case).

    Returns
    -------
    float
        Tau [s].
    """
    attrs = VEHICLE_ATTRIBUTES.get(vehicle.lower())
    if attrs is not None:
        return attrs[1]
    return DEFAULT_TAU
