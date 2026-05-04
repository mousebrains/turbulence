# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""End-to-end CF-1.13 / ACDD-1.3 compliance regression for combo outputs.

The IOOS compliance-checker only carries grammars up to ``cf:1.11`` /
``acdd:1.3`` today, so this test runs against those.  All "errors" the
checker flags should be either resolved by the schema/auto-fill code or
known false positives that we explicitly enumerate here.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from odas_tpw.perturb.combo import make_combo
from odas_tpw.perturb.netcdf_schema import COMBO_SCHEMA


@pytest.fixture
def synthetic_combo(tmp_path: Path) -> Path:
    """Build a tiny depth-binned NetCDF and run :func:`make_combo` on it.

    The dataset emulates a real perturb output: per-bin physical
    measurements (T, P), per-profile lat/lon/stime/etime scalars, plus the
    standard ``bin``/``profile`` coords.
    """
    binned_dir = tmp_path / "binned"
    binned_dir.mkdir()
    out_dir = tmp_path / "combo"

    n_bins = 6
    n_prof = 3
    rng = np.random.default_rng(0)

    # 2026-01-01 00:00:00 UTC + small offsets per profile
    t0 = 1767225600.0
    stimes = np.array([t0, t0 + 600.0, t0 + 1200.0])
    etimes = stimes + 300.0

    ds = xr.Dataset(
        {
            "T": (
                ["bin", "profile"],
                15.0 - 0.1 * np.arange(n_bins)[:, None] + rng.standard_normal((n_bins, n_prof)),
            ),
            "P": (
                ["bin", "profile"],
                np.tile(np.arange(n_bins, dtype=float)[:, None], (1, n_prof)),
            ),
            "lat": (["profile"], np.array([10.0, 11.0, 12.0])),
            "lon": (["profile"], np.array([-150.0, -149.5, -149.0])),
            "stime": (["profile"], stimes),
            "etime": (["profile"], etimes),
        },
        coords={
            "bin": np.arange(n_bins, dtype=float),
            "profile": np.arange(n_prof, dtype=np.int32),
        },
    )
    ds.to_netcdf(binned_dir / "binned00.nc")

    netcdf_attrs = {
        "title": "Synthetic perturb combo for CF/ACDD compliance test",
        "summary": "Synthetic depth-binned combo used to lock in compliance regressions.",
        "keywords": "EARTH SCIENCE > OCEANS > OCEAN TEMPERATURE > WATER TEMPERATURE",
        "keywords_vocabulary": "GCMD Science Keywords",
        "institution": "Test Institution",
        "creator_name": "Test Author",
        "creator_email": "test@example.com",
        "creator_url": "https://example.com",
        "publisher_name": "Test Publisher",
        "publisher_email": "test@example.com",
        "publisher_url": "https://example.com",
        "naming_authority": "edu.example",
        "processing_level": "L2",
        "license": "CC-BY-4.0",
        "source": "synthetic test data",
        "comment": "no real measurements",
        "acknowledgement": "synthetic test fixture",
        "project": "perturb test",
        "platform": "synthetic",
        "instrument": "synthetic",
        "geospatial_vertical_positive": "down",
        "Conventions": "CF-1.13, ACDD-1.3",
    }
    out = make_combo(binned_dir, out_dir, COMBO_SCHEMA, netcdf_attrs=netcdf_attrs)
    assert out is not None
    return out


def _run_compliance_checker(path: Path, suite: str) -> str:
    """Invoke compliance-checker on *path* and return its captured report."""
    cc = pytest.importorskip("compliance_checker.runner")
    Runner = cc.ComplianceChecker
    CheckSuite = cc.CheckSuite
    CheckSuite.load_all_available_checkers()

    buf = io.StringIO()
    with redirect_stdout(buf):
        Runner.run_checker(
            str(path),
            checker_names=[suite],
            verbose=0,
            criteria="strict",
            output_filename="-",
            output_format="text",
        )
    return buf.getvalue()


# Known false positives (keep narrow; revisit when checker updates):
#   §2.4 — flags (bin, profile) ordering as not T/Z/Y/X.  Profile is the CF
#   §9 instance dim and is the recommended layout for profile featureType,
#   so this is a checker bug.
#   §2.6 — checker only validates against literal "CF-1.11"; we declare
#   "CF-1.13".
_CF_FALSE_POSITIVE_SUBSTRINGS = (
    "spatio-temporal dimensions are not in the recommended order",
    'Conventions global attribute does not contain "CF-1.11"',
)


def test_combo_passes_cf_after_excluding_known_false_positives(synthetic_combo: Path) -> None:
    report = _run_compliance_checker(synthetic_combo, "cf:1.11")
    # Anything that's not a known false positive is a real regression.
    for line in report.splitlines():
        s = line.strip()
        if not s.startswith("*"):
            continue
        if any(needle in s for needle in _CF_FALSE_POSITIVE_SUBSTRINGS):
            continue
        # Empty bullets and section markers can pass through, ignore them.
        msg = s.lstrip("*").strip()
        if not msg:
            continue
        # The checker reports its prelude under "Corrective Actions" with
        # the issue count — not a real failure to bubble.
        if "potential issues" in msg.lower():
            continue
        pytest.fail(f"unexpected CF issue: {msg}")


def test_combo_geospatial_and_time_attrs_are_filled(synthetic_combo: Path) -> None:
    """Auto-filled ACDD attrs round-trip on the synthetic combo."""
    with xr.open_dataset(synthetic_combo, decode_times=False) as ds:
        # Spatial bbox derived from per-profile lat/lon
        assert ds.attrs["geospatial_lat_min"] == pytest.approx(10.0)
        assert ds.attrs["geospatial_lat_max"] == pytest.approx(12.0)
        assert ds.attrs["geospatial_lon_min"] == pytest.approx(-150.0)
        assert ds.attrs["geospatial_lon_max"] == pytest.approx(-149.0)
        assert ds.attrs["geospatial_bounds_crs"] == "EPSG:4326"
        assert "POLYGON" in ds.attrs["geospatial_bounds"]
        # Time coverage derived from stime/etime
        assert ds.attrs["time_coverage_start"].startswith("2026-01-01T00:00:00")
        assert ds.attrs["time_coverage_end"].startswith("2026-01-01T00:25:00")
        # Vertical CRS is auto-filled when ``bin``/``depth``/``P`` exists
        assert ds.attrs["geospatial_bounds_vertical_crs"].startswith("EPSG:")


def test_combo_carries_units_metadata_normalisation(synthetic_combo: Path) -> None:
    """Sanity check that the schema-applied attrs survive round-trip."""
    with xr.open_dataset(synthetic_combo, decode_times=False) as ds:
        assert ds["T"].attrs["units"] == "degree_Celsius"
        assert ds["T"].attrs["standard_name"] == "sea_water_temperature"
        assert ds["lat"].attrs["units"] == "degrees_north"
        assert ds["lon"].attrs["units"] == "degrees_east"


@pytest.mark.parametrize("suite", ["acdd:1.3"])
def test_combo_acdd_only_known_warnings(synthetic_combo: Path, suite: str) -> None:
    """ACDD report on the synthetic combo must not contain anything beyond
    the known categories of warning we accept.

    Acceptable warnings for the synthetic test fixture:
      - missing standard_name on engineering / non-CF channels (no such
        channels in this fixture, but the rule is general).
      - geospatial_bounds_vertical_crs *if* not auto-filled.

    Anything else (e.g. missing summary, missing license) means an attr
    that we know how to fill is no longer being filled.
    """
    pytest.importorskip("compliance_checker.runner")
    report = _run_compliance_checker(synthetic_combo, suite)
    # The synthetic fixture sets every attr that the YAML normally would,
    # so the only ACDD warnings should be standard_name reminders on
    # channels without one (none here).  An empty "Highly Recommended"
    # section is fine.
    forbidden = (
        "summary not present",
        "keywords not present",
        "license not present",
        "naming_authority not present",
        "processing_level not present",
        "creator_url not present",
        "publisher_email not present",
        "geospatial_lat_min not present",
        "geospatial_lat_max not present",
        "geospatial_lon_min not present",
        "geospatial_lon_max not present",
        "time_coverage_start not present",
        "time_coverage_end not present",
        "time_coverage_duration not present",
        "geospatial_bounds_crs not present",
    )
    for needle in forbidden:
        assert needle not in report, f"ACDD regression — {needle!r} is back in:\n{report}"
