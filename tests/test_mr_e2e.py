# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""End-to-end test of the batch pipeline on real MicroRider-on-glider data.

This is the first test that runs the batch machinery (PFile -> vehicle
resolution -> profile detection -> epsilon -> chi) against a real glider
cast (issue #131 m11).  Every prior MR/glider test was synthetic or mocked,
which is how glider-hostile defaults could ship with all tests passing.

The fixture, ``tests/data/MR_SL685_climb.p``, is a 150 s climb segment cut
at record boundaries (``extract_pfile_segment``, records 120-269) from
``MR/AIOP2_SL685_0450.p``: an MR1000RDL-EM SN 435 on Slocum glider osu685
(ARCTERX IOP2, Feb 2025) ascending steadily from ~481 to ~429 dbar at
~0.35 dbar/s, with the EM flowmeter reading ~0.5 m/s through-water speed.

Direction, W_min, and speed are passed EXPLICITLY (direction="glide",
W_min=0.05, speed=median |U_EM|) so the test is independent of the
vehicle-resolved defaults: it must pass on main today and keep passing
after the glider-aware defaults land.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from odas_tpw.rsi import PFile
from odas_tpw.rsi.chi_io import compute_chi_file
from odas_tpw.rsi.dissipation import compute_diss_file
from odas_tpw.rsi.profile import _smooth_fall_rate
from odas_tpw.rsi.vehicle import resolve_direction
from odas_tpw.scor160.profile import get_profiles

FIXTURE = Path(__file__).parent / "data" / "MR_SL685_climb.p"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="real-MR fixture tests/data/MR_SL685_climb.p not present",
)

# Slocum-glider smoothing time constant (vehicle.py VEHICLE_ATTRIBUTES);
# passed explicitly so the test does not depend on default resolution.
GLIDER_TAU = 3.0


@pytest.fixture(scope="module")
def pf() -> PFile:
    return PFile(FIXTURE)


@pytest.fixture(scope="module")
def em_speed(pf: PFile) -> float:
    """Median through-water speed from the EM flowmeter [m/s]."""
    return float(np.median(np.abs(pf.channels["U_EM"])))


@pytest.fixture(scope="module")
def eps_ds(pf: PFile, em_speed: float, tmp_path_factory: pytest.TempPathFactory) -> xr.Dataset:
    """Epsilon product computed once for the module (explicit fixed speed)."""
    out = tmp_path_factory.mktemp("mr_eps")
    paths = compute_diss_file(FIXTURE, out, speed=em_speed, direction="glide")
    assert len(paths) >= 1, "compute_diss_file wrote no epsilon output"
    with xr.open_dataset(paths[0]) as ds:
        return ds.load()


@pytest.fixture(scope="module")
def chi_ds(pf: PFile, em_speed: float, tmp_path_factory: pytest.TempPathFactory) -> xr.Dataset:
    """Chi (Method 2) product computed once for the module."""
    out = tmp_path_factory.mktemp("mr_chi")
    paths = compute_chi_file(FIXTURE, out, speed=em_speed, direction="glide")
    assert len(paths) >= 1, "compute_chi_file wrote no chi output"
    with xr.open_dataset(paths[0]) as ds:
        return ds.load()


# ---------------------------------------------------------------------------
# Fixture integrity: a real glider-MR file with a genuine pressure ramp
# ---------------------------------------------------------------------------


def test_fixture_is_glider_microrider(pf: PFile) -> None:
    info = pf.config["instrument_info"]
    assert info.get("vehicle", "").lower() == "slocum_glider"
    assert info.get("model") == "MR1000RDL-EM"
    assert info.get("sn") == "435"
    assert pf.fs_fast == pytest.approx(512.0, abs=1.0)
    assert pf.fs_slow == pytest.approx(64.0, abs=1.0)


def test_fixture_has_mr_channels(pf: PFile) -> None:
    for name in ("P", "U_EM", "Incl_X", "Incl_Y", "sh1", "sh2", "T1_dT1", "T2_dT2", "Ax", "Ay"):
        assert name in pf.channels, f"channel {name} missing from fixture"


def test_fixture_pressure_ramp(pf: PFile) -> None:
    """A genuine, monotonic-ish climb: ~52 dbar over ~150 s."""
    P = pf.channels["P"]
    duration = float(pf.t_slow[-1] - pf.t_slow[0])
    assert 100.0 < duration < 200.0
    assert P[0] > 470.0
    assert P[-1] < 440.0
    assert P[0] - P[-1] > 40.0  # real ramp, not jitter
    # Ascending nearly everywhere at the slow rate
    assert np.mean(np.diff(P) < 0) > 0.9


def test_em_speed_plausible(em_speed: float) -> None:
    """EM flowmeter reads a plausible Slocum through-water speed."""
    assert 0.2 < em_speed < 1.0


# ---------------------------------------------------------------------------
# Vehicle resolution + profile detection (explicit glide / W_min)
# ---------------------------------------------------------------------------


def test_vehicle_resolves_to_glide(pf: PFile) -> None:
    vehicle = pf.config["instrument_info"].get("vehicle", "").lower()
    assert resolve_direction("auto", vehicle) == "glide"


def test_profile_detection_finds_climb(pf: PFile) -> None:
    P = pf.channels["P"]
    W = _smooth_fall_rate(P, pf.fs_slow, tau=GLIDER_TAU)
    profiles = get_profiles(P, W, pf.fs_slow, W_min=0.05, direction="glide")
    assert len(profiles) >= 1
    # The dominant profile covers most of the segment and ascends
    start, end = max(profiles, key=lambda p: p[1] - p[0])
    assert (end - start) / pf.fs_slow > 100.0
    assert P[start] > P[end]


# ---------------------------------------------------------------------------
# Epsilon end-to-end (explicit fixed speed from the EM flowmeter)
# ---------------------------------------------------------------------------


def test_epsilon_finite_and_plausible(eps_ds: xr.Dataset) -> None:
    eps = eps_ds["epsilon"].values
    assert eps.size > 0
    assert np.all(np.isfinite(eps))
    # Every estimate physically sane; the bulk in the plausible open-ocean
    # range for a quiet 430-480 dbar interior segment.
    assert np.all(eps > 1e-14)
    assert np.all(eps < 1e-4)
    assert 1e-12 < np.median(eps) < 1e-5


def test_epsilon_product_attrs(eps_ds: xr.Dataset, em_speed: float) -> None:
    for attr in ("fft_length", "diss_length", "fs_fast", "source", "instrument", "sn"):
        assert attr in eps_ds.attrs, f"epsilon product missing attr {attr}"
    assert eps_ds.attrs["instrument"] == "MR1000RDL-EM"
    assert str(eps_ds.attrs["sn"]) == "435"
    # The explicit fixed speed is carried into the product
    assert float(eps_ds["speed"].min()) == pytest.approx(em_speed, rel=1e-6)
    assert float(eps_ds["speed"].max()) == pytest.approx(em_speed, rel=1e-6)
    # Window mean pressures fall inside the fixture's ramp
    assert float(eps_ds["P_mean"].min()) > 420.0
    assert float(eps_ds["P_mean"].max()) < 490.0


# ---------------------------------------------------------------------------
# Chi (Method 2) end-to-end
# ---------------------------------------------------------------------------


def test_chi_finite_and_plausible(chi_ds: xr.Dataset) -> None:
    chi = chi_ds["chi"].values
    assert chi.size > 0
    assert np.all(np.isfinite(chi))
    assert np.all(chi > 1e-13)
    assert np.all(chi < 1e-5)
    assert 1e-11 < np.median(chi) < 1e-7


def test_chi_product_attrs(chi_ds: xr.Dataset) -> None:
    for attr in ("fft_length", "diss_length", "spectrum_model", "chi_method", "source", "sn"):
        assert attr in chi_ds.attrs, f"chi product missing attr {attr}"
    assert chi_ds.attrs["chi_method"] == "fit"  # Method 2: no epsilon input
    assert float(chi_ds["P_mean"].min()) > 420.0
    assert float(chi_ds["P_mean"].max()) < 490.0
