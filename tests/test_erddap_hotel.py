# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the ERDDAP -> hotel builder.

None of these need a network. The two parts most likely to be silently wrong
are the query builder (pure strings) and the QC layer (takes a Dataset, returns
one), and both are testable by construction; the fetcher's failure modes are
covered against a real ``http.server`` on localhost serving recorded-shaped
bodies, per docs/erddap_access_DESIGN.md section 9.
"""

from __future__ import annotations

import http.server
import itertools
import threading
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import xarray as xr

from odas_tpw.erddap.build import (
    DEFAULT_FILL,
    build,
    mask_fill_values,
    plan_requests,
)
from odas_tpw.erddap.config import DEFAULTS, load_config, to_builder_config, validate
from odas_tpw.erddap.fetch import (
    EmptyWindow,
    ErddapError,
    cache_path,
    count_rows,
    das_fingerprint,
    fetch_bytes,
    normalize_das,
    recall_das_sha,
    remember_das_sha,
)
from odas_tpw.erddap.query import chunk_windows, iso, tabledap_url

REPO_ROOT = Path(__file__).resolve().parents[1]

T0 = 1633046400.0  # 2021-10-01T00:00:00Z


# --------------------------------------------------------------- query builder
def test_iso_is_the_form_erddap_wants():
    assert iso(T0) == "2021-10-01T00:00:00Z"


def test_chunk_windows_tile_without_gap_or_overlap():
    wins = chunk_windows(T0, T0 + 10 * 86400, 3)
    assert len(wins) == 4
    assert wins[0][0] == T0
    assert wins[-1][1] == T0 + 10 * 86400
    for a, b in itertools.pairwise(wins):
        # Half-open: one window ends exactly where the next begins, so no row
        # is fetched twice and none falls between two chunks.
        assert a[1] == b[0]


@pytest.mark.parametrize("bad", [0, -1])
def test_chunk_windows_rejects_a_nonpositive_step(bad):
    with pytest.raises(ValueError, match="chunk_days"):
        chunk_windows(T0, T0 + 86400, bad)


def test_chunk_windows_rejects_an_inverted_range():
    with pytest.raises(ValueError, match="must be after"):
        chunk_windows(T0 + 86400, T0, 1)


def test_url_percent_encodes_the_whole_constraint():
    """`>=`, `,` and the `:` in a timestamp must all be encoded.

    ERDDAP accepts some unencoded forms and not others, and the failures are
    400s that read like a missing variable rather than a quoting bug.
    """
    url = tabledap_url(
        "https://example.org/erddap",
        "ds-raw",
        ["sci_ctd41cp_timestamp", "sci_water_temp"],
        window=(T0, T0 + 86400),
        constraints=["distinct()"],
    )
    assert url == (
        "https://example.org/erddap/tabledap/ds-raw.nc"
        "?sci_ctd41cp_timestamp%2Csci_water_temp"
        "&time%3E%3D2021-10-01T00%3A00%3A00Z"
        "&time%3C2021-10-02T00%3A00%3A00Z"
        "&distinct%28%29"
    )
    # Only the ? and the joining & stay literal.
    assert url.count("?") == 1
    assert ">" not in url and "," not in url.split("?", 1)[1]


def test_last_window_is_inclusive_so_the_final_row_is_not_dropped():
    mid = tabledap_url("https://e.org", "d", ["time"], window=(T0, T0 + 60))
    end = tabledap_url("https://e.org", "d", ["time"], window=(T0, T0 + 60), last_window=True)
    assert "%3C2021" in mid and "%3C%3D2021" not in mid
    assert "%3C%3D2021" in end


def test_base_url_may_or_may_not_already_end_in_tabledap():
    a = tabledap_url("https://e.org/erddap", "d", ["time"])
    b = tabledap_url("https://e.org/erddap/tabledap", "d", ["time"])
    assert a == b


def test_url_refuses_an_empty_variable_list():
    with pytest.raises(ValueError, match="at least one"):
        tabledap_url("https://e.org", "d", [])


# ----------------------------------------------------------------- .das digest
_DAS_HEAD = 'Attributes {\n  NC_GLOBAL {\n    String date_modified "2021-11-10T20:11:47Z";\n'


def _das(stamp: str) -> str:
    """A .das with ERDDAP's per-request lines stamped into its own history."""
    return (
        _DAS_HEAD
        + '    String history "2021-11-10T20:11:47Z: /home/kerfoot/proc.py in.dat\n'
        + f"{stamp} (local files)\n"
        + f'{stamp} http://example.org/erddap/tabledap/d.das";\n  }}\n}}\n'
    )


def test_das_digest_ignores_the_per_request_lines_erddap_injects():
    """The bug this guards: hashing the .das verbatim never hits the cache.

    ERDDAP stamps the time YOU ASKED into the .das's own history, so two
    fetches a second apart differ. Measured on Rutgers' live server. Hashing
    that would make every run a cache miss and make `verify` report CHANGED on
    an untouched dataset.
    """
    a, b = _das("2026-08-24T02:09:43Z"), _das("2026-08-24T02:09:44Z")
    assert a != b, "fixture should differ, as the real bodies do"
    assert das_fingerprint(a)[0] == das_fingerprint(b)[0]


def test_das_digest_still_changes_when_the_dataset_really_does():
    a = _das("2026-08-24T02:09:43Z")
    b = a.replace("2021-11-10T20:11:47Z", "2022-01-01T00:00:00Z")
    assert das_fingerprint(a)[0] != das_fingerprint(b)[0]


def test_das_normalisation_keeps_upstream_provenance():
    """The upstream lines have a colon after the Z; ERDDAP's request log does not."""
    kept = normalize_das(_das("2026-08-24T02:09:43Z"))
    assert "/home/kerfoot/proc.py" in kept
    assert "(local files)" not in kept


def test_das_fingerprint_reads_date_modified():
    assert das_fingerprint(_das("2026-08-24T02:09:43Z"))[1] == "2021-11-10T20:11:47Z"


def test_cache_key_changes_with_the_dataset_revision(tmp_path):
    """A server-side reprocess must miss the cache by construction."""
    url = "https://e.org/erddap/tabledap/d.nc?time"
    assert cache_path(tmp_path, "d", url, "sha-a") != cache_path(tmp_path, "d", url, "sha-b")
    assert cache_path(tmp_path, "d", url, "sha-a") == cache_path(tmp_path, "d", url, "sha-a")


# -------------------------------------------------------------- _FillValue QC
def _ds(**cols) -> xr.Dataset:
    return xr.Dataset({k: (("row",), np.asarray(v, dtype=float)) for k, v in cols.items()})


def test_fill_value_is_masked_even_though_it_arrives_finite():
    """9.96921e+36 is NOT masked on read; it comes through as a real float."""
    ds = _ds(sci_water_temp=[20.0, DEFAULT_FILL, 21.0])
    stats = mask_fill_values(ds, ["sci_water_temp"])
    assert stats["sci_water_temp"]["fill_masked"] == 1
    assert np.isnan(ds["sci_water_temp"].values[1])
    assert ds["sci_water_temp"].values[0] == 20.0


def test_fill_value_matches_despite_float32_widening():
    """`== 9.96921e+36` misses it: the value is served float32 and widened."""
    ds = _ds(t=[float(np.float32(DEFAULT_FILL)), 20.0])
    mask_fill_values(ds, ["t"])
    assert np.isnan(ds["t"].values[0])
    assert ds["t"].values[1] == 20.0


def test_a_declared_fill_value_wins_over_the_conventional_one():
    ds = _ds(t=[20.0, -9999.0])
    ds["t"].attrs["_FillValue"] = -9999.0
    stats = mask_fill_values(ds, ["t"])
    assert stats["t"]["fill_masked"] == 1
    assert np.isnan(ds["t"].values[1])


def test_zero_is_kept_by_default():
    """A real 0 degC sample is polar water, not a sentinel.

    The notebook's blanket `0.0 -> NaN` would delete it. Measured over 5.4M
    rows, the timestamp bounds already remove every genuine sentinel.
    """
    ds = _ds(sci_water_temp=[0.0, 20.0])
    stats = mask_fill_values(ds, ["sci_water_temp"])
    assert stats["sci_water_temp"]["zero_masked"] == 0
    assert ds["sci_water_temp"].values[0] == 0.0


def test_zero_is_dropped_only_when_a_variable_opts_in():
    ds = _ds(a=[0.0, 20.0], b=[0.0, 20.0])
    stats = mask_fill_values(ds, ["a", "b"], drop_zero=["a"])
    assert stats["a"]["zero_masked"] == 1 and np.isnan(ds["a"].values[0])
    assert stats["b"]["zero_masked"] == 0 and ds["b"].values[0] == 0.0


# --------------------------------------------------------------------- config
def _cfg(**over) -> dict:
    cfg = {
        "server": {"base_url": "https://e.org/erddap", "dataset_id": "d-raw"},
        "fetch": {
            "variables": ["sci_ctd41cp_timestamp", "sci_water_temp"],
            "time_min": "2021-10-01T00:00:00Z",
            "time_max": "2021-10-03T00:00:00Z",
            "chunk_days": 1,
        },
        "qc": {"time_base": "sci_ctd41cp_timestamp"},
        "time": {"base": "sci_ctd41cp_timestamp", "max_gap": 30.0},
        "sensors": {"sci_water_temp": {"units": "degree_Celsius"}},
        "output": {"file": "hotel.nc"},
    }
    for section, values in over.items():
        cfg.setdefault(section, {}).update(values)
    return cfg


def test_a_minimal_config_validates():
    validate(_cfg())


def test_the_clock_must_be_downloaded_like_any_other_column():
    cfg = _cfg()
    cfg["fetch"]["variables"] = ["sci_water_temp"]
    with pytest.raises(ValueError, match=r"not in fetch\.variables"):
        validate(cfg)


def test_griddap_is_refused_rather_than_half_working():
    with pytest.raises(ValueError, match="only 'tabledap'"):
        validate(_cfg(server={"protocol": "griddap"}))


@pytest.mark.parametrize(
    ("section", "values", "match"),
    [
        ("fetch", {"refresh": "sometimes"}, "refresh"),
        ("fetch", {"format": "parquet"}, "format"),
        ("fetch", {"chunk_days": 0}, "chunk_days"),
        ("fetch", {"overlap_chunks": -1}, "overlap_chunks"),
        ("qc", {"dedupe": "median"}, "dedupe"),
        ("time", {"max_gap": 0}, "max_gap"),
        ("server", {"timeout_s": 0}, "timeout_s"),
    ],
)
def test_bad_values_are_named(section, values, match):
    with pytest.raises(ValueError, match=match):
        validate(_cfg(**{section: values}))


def test_a_valid_range_on_an_undownloaded_variable_is_refused():
    """Silently doing nothing is worse than refusing: the user thinks it applied."""
    with pytest.raises(ValueError, match="never downloaded"):
        validate(_cfg(qc={"valid_range": {"sci_water_cond": [0, 10]}}))


def test_an_inverted_valid_range_is_refused():
    with pytest.raises(ValueError, match="valid_range"):
        validate(_cfg(qc={"valid_range": {"sci_water_temp": [40, -5]}}))


def test_an_unknown_sensor_option_is_named():
    with pytest.raises(ValueError, match="unknown option"):
        validate(_cfg(sensors={"sci_water_temp": {"unts": "degC"}}))


def test_valid_range_becomes_the_builders_per_sensor_bounds():
    cfg = _cfg(qc={"valid_range": {"sci_water_temp": [-5.0, 40.0]}})
    built = to_builder_config(cfg)
    assert built["sensors"]["sci_water_temp"]["valid_min"] == -5.0
    assert built["sensors"]["sci_water_temp"]["valid_max"] == 40.0
    # The clock is the time base, not a channel.
    assert "sci_ctd41cp_timestamp" not in built["sensors"]


def test_the_gap_limit_survives_the_translation():
    """time.max_gap here is projection.max_gap there; losing it in the mapping
    would silently restore the fabricated-ramp behaviour #150 removed."""
    built = to_builder_config(_cfg())
    assert built["projection"]["max_gap"] == 30.0
    assert built["projection"]["extrapolate"] is False


def test_an_explicit_sensor_bound_is_not_overwritten_by_valid_range():
    cfg = _cfg(
        qc={"valid_range": {"sci_water_temp": [-5.0, 40.0]}},
        sensors={"sci_water_temp": {"valid_min": 0.0, "valid_max": 30.0}},
    )
    built = to_builder_config(cfg)
    assert built["sensors"]["sci_water_temp"]["valid_min"] == 0.0


# --------------------------------------------------------- shipped artifacts
def test_the_shipped_example_loads_and_validates():
    """examples/rutgers_erddap/erddap-hotel.yaml is the schema's reference."""
    example = REPO_ROOT / "examples/rutgers_erddap/erddap-hotel.yaml"
    if not example.exists():
        pytest.skip("example not present")
    cfg = load_config(example)
    validate(cfg)
    to_builder_config(cfg)


def test_the_generated_template_loads_and_validates(tmp_path):
    """`init` must not emit something `build` then rejects."""
    from odas_tpw.erddap.config import generate_template

    path = generate_template(tmp_path / "erddap-hotel.yaml")
    cfg = load_config(path)
    validate(cfg)


def test_every_defaults_section_is_documented_in_the_template(tmp_path):
    from odas_tpw.erddap.config import generate_template

    text = generate_template(tmp_path / "t.yaml").read_text()
    for section in DEFAULTS:
        assert f"\n{section}:" in text, f"template omits the {section!r} section"


def test_plan_requests_needs_a_time_min():
    """Without one the whole dataset is requested (design F5)."""
    cfg = _cfg()
    cfg["fetch"]["time_min"] = None
    with pytest.raises(ValueError, match="time_min"):
        plan_requests(cfg)


def test_plan_requests_is_deterministic_and_covers_the_window():
    plan = plan_requests(_cfg())
    assert [p["start"] for p in plan] == ["2021-10-01T00:00:00Z", "2021-10-02T00:00:00Z"]
    assert plan[-1]["end"] == "2021-10-03T00:00:00Z"
    assert plan == plan_requests(_cfg()), "same config must give the same URLs"


def test_plan_requests_accepts_epoch_seconds_as_well_as_iso():
    cfg = _cfg()
    cfg["fetch"]["time_min"] = T0
    cfg["fetch"]["time_max"] = T0 + 86400
    assert plan_requests(cfg)[0]["start"] == "2021-10-01T00:00:00Z"


def test_plan_requests_rejects_an_unparseable_date():
    cfg = _cfg()
    cfg["fetch"]["time_min"] = "last tuesday"
    with pytest.raises(ValueError, match="ISO-8601"):
        plan_requests(cfg)


# ------------------------------------------------------- fetcher, fake server
class _Handler(http.server.BaseHTTPRequestHandler):
    # BaseHTTPRequestHandler is instantiated per request, so the routing table
    # and the hit counter have to live on the class.
    routes: ClassVar[dict[str, tuple[int, bytes, str]]] = {}
    hits: ClassVar[dict[str, int]] = {}

    def do_GET(self):
        key = self.path.split("?", 1)[0]
        self.hits[key] = self.hits.get(key, 0) + 1
        status, body, ctype = self.routes.get(key, (404, b"not found", "text/plain"))
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence the test run
        pass


@pytest.fixture
def server():
    _Handler.routes, _Handler.hits = {}, {}
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", _Handler
    httpd.shutdown()
    httpd.server_close()


def test_an_empty_time_window_is_not_an_error(server):
    """F6b: ERDDAP 404s a window with no rows, the same status as a bad ID.

    A gap in a deployment is data. Treating it as a failure would abort a
    build over a mission the glider spent on the surface.
    """
    base, handler = server
    handler.routes["/empty"] = (
        404,
        b'Error {\n  code=404;\n  message="Your query produced no matching results.";\n}',
        "text/plain",
    )
    with pytest.raises(EmptyWindow):
        fetch_bytes(f"{base}/empty", retries=0)
    assert count_rows(f"{base}/empty", retries=0) == 0


def test_a_400_is_not_retried(server):
    """A malformed variable list is a bug; asking again asks the same wrong question."""
    base, handler = server
    handler.routes["/bad"] = (
        400,
        b'Error {\n  message="Unrecognized variable=sci_water_tmep";\n}',
        "text/plain",
    )
    with pytest.raises(ErddapError, match="Unrecognized variable"):
        fetch_bytes(f"{base}/bad", retries=3, _sleep=lambda _s: None)
    assert handler.hits["/bad"] == 1


def test_a_500_is_retried_then_reported(server):
    base, handler = server
    handler.routes["/boom"] = (500, b"Error {\n  message=\"internal\";\n}", "text/plain")
    with pytest.raises(ErddapError, match="gave up after 2 retries"):
        fetch_bytes(f"{base}/boom", retries=2, _sleep=lambda _s: None)
    assert handler.hits["/boom"] == 3


def test_a_404_that_is_not_the_empty_message_stays_an_error(server):
    """A wrong dataset ID and an empty window share a status; only the text differs."""
    base, handler = server
    handler.routes["/nope"] = (
        404,
        b'Error {\n  message="Unknown datasetID=typo";\n}',
        "text/plain",
    )
    with pytest.raises(ErddapError, match="Unknown datasetID"):
        fetch_bytes(f"{base}/nope", retries=0)


def test_a_200_that_is_not_netcdf_is_refused(server, tmp_path):
    """F1: a proxy or captive portal answers 200 with HTML. It must not cache."""
    from odas_tpw.erddap.fetch import fetch_to_file

    base, handler = server
    handler.routes["/html"] = (200, b"<html><body>Sign in to continue</body></html>", "text/html")
    dest = tmp_path / "chunk.nc"
    with pytest.raises(ErddapError, match="not NetCDF"):
        fetch_to_file(f"{base}/html", dest, retries=0)
    assert not dest.exists()


def test_a_truncated_netcdf_is_refused_and_leaves_no_partial(server, tmp_path):
    """F3: served chunked with no Content-Length, so a cut is invisible by length.

    The body still starts with CDF\\x01, so only opening it catches this.
    """
    from odas_tpw.erddap.fetch import fetch_to_file

    base, handler = server
    # A real NetCDF cut in half -- a hand-made stub of nulls is not enough,
    # netCDF4 accepts it.
    whole = tmp_path / "whole.nc"
    xr.Dataset({"t": (("row",), np.arange(500.0))}).to_netcdf(whole)
    handler.routes["/cut"] = (200, whole.read_bytes()[:120], "application/x-netcdf")
    dest = tmp_path / "chunk.nc"
    with pytest.raises(ErddapError, match="will not open"):
        fetch_to_file(f"{base}/cut", dest, retries=0)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_count_rows_discounts_the_two_header_lines(server):
    """ERDDAP's CSV carries a name row and a units row."""
    base, handler = server
    handler.routes["/csv"] = (
        200,
        b"time\nUTC\n2021-10-01T00:00:00Z\n2021-10-01T00:00:01Z\n",
        "text/csv",
    )
    assert count_rows(f"{base}/csv", retries=0) == 2


# ------------------------------------------------------- offline end-to-end
def test_build_offline_produces_a_hotel_file(tmp_path):
    """The whole chain with the network replaced by a pre-seeded cache.

    Exercises what a real build does after the bytes land: fill masking,
    sanitising, projection onto the CTD clock, unit scaling, and the write.
    """
    from odas_tpw.erddap.build import fetch_chunks

    cfg = _cfg(
        server={"cache": str(tmp_path / "cache")},
        fetch={
            "variables": ["sci_ctd41cp_timestamp", "sci_water_temp", "sci_water_cond"],
            "time_min": T0,
            "time_max": T0 + 86400,
            "chunk_days": 1,
            "refresh": "never",
        },
        qc={"time_min": T0 - 86400, "time_max": T0 + 2 * 86400},
        sensors={
            "sci_water_temp": {"units": "degree_Celsius"},
            "sci_water_cond": {"units": "mS/cm", "scale": 10.0},
        },
        output={"file": str(tmp_path / "hotel.nc")},
    )
    # A chunk shaped like a real one: a fill value, a duplicate stamp, an
    # out-of-order row, and conductivity in the S/m the server serves.
    t = np.array([T0 + 3, T0 + 1, T0 + 2, T0 + 2, DEFAULT_FILL, T0 + 4])
    ds = xr.Dataset(
        {
            "sci_ctd41cp_timestamp": (("row",), t),
            "sci_water_temp": (("row",), np.array([20.3, 20.1, 20.2, 20.25, 99.0, 20.4])),
            "sci_water_cond": (("row",), np.array([4.44, 4.41, 4.42, 4.43, 9.9, 4.45])),
        }
    )
    ds.attrs["history"] = "2021-11-10T20:11:47Z: upstream proc.py"

    # Seed the cache the way an ONLINE run does: under a real .das digest,
    # with the sidecar recording it. Seeding under whatever key offline happens
    # to use would make this test agree with a bug rather than check for one --
    # which is exactly what it did before, when offline substituted the literal
    # string "offline" into the key and so could never find an online cache.
    das_sha = "0123456789abcdef" * 4
    remember_das_sha(tmp_path / "cache", "d-raw", das_sha)
    plan = plan_requests(cfg)
    dest = cache_path(tmp_path / "cache", "d-raw", plan[0]["url"], das_sha)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dest)

    paths, meta = fetch_chunks(cfg, now=T0, offline=True)
    assert len(paths) == 1 and meta["chunks_cached"] == 1

    out = build(cfg, now=T0, offline=True)
    with xr.open_dataset(out, decode_times=False) as got:
        times = np.asarray(got["sci_ctd41cp_timestamp"].values)
        assert np.all(np.diff(times) > 0), "the output clock must be strictly increasing"
        assert times.size == 4, "the fill-value stamp is dropped, the duplicate collapsed"
        # S/m -> mS/cm applied exactly once, in the builder.
        assert np.nanmax(got["sci_water_cond"].values) == pytest.approx(44.5, abs=0.2)
        assert got["sci_water_cond"].attrs["units"] == "mS/cm"
        assert "upstream proc.py" in got.attrs["history"]
        assert "erddap-hotel build" in got.attrs["history"]
        assert got.attrs["erddap_dataset_id"] == "d-raw"


def test_build_offline_refuses_when_a_chunk_is_not_cached(tmp_path):
    cfg = _cfg(
        server={"cache": str(tmp_path / "cache")},
        fetch={"refresh": "never"},
        output={"file": str(tmp_path / "hotel.nc")},
    )
    remember_das_sha(tmp_path / "cache", "d-raw", "deadbeef" * 8)
    with pytest.raises(ErddapError, match="not cached"):
        build(cfg, now=T0, offline=True)


def test_build_offline_says_so_when_the_cache_was_never_populated(tmp_path):
    """Distinct from 'this chunk is missing': there is nothing here at all."""
    cfg = _cfg(
        server={"cache": str(tmp_path / "cache")},
        output={"file": str(tmp_path / "hotel.nc")},
    )
    with pytest.raises(ErddapError, match="never been populated"):
        build(cfg, now=T0, offline=True)


def test_offline_reuses_the_digest_the_cache_was_populated_under(tmp_path):
    """The bug this guards: offline could never find an online-filled cache.

    The cache key includes the .das digest. Offline cannot compute one, and
    substituting a placeholder means looking under a key no online run ever
    wrote -- every lookup missed while the data sat right there. Verified
    against the live server before the fix.
    """
    sha = "a" * 64
    remember_das_sha(tmp_path / "cache", "d-raw", sha)
    assert recall_das_sha(tmp_path / "cache", "d-raw") == sha
    # The path offline resolves must be the one an online run wrote.
    url = "https://e.org/erddap/tabledap/d-raw.nc?time"
    assert cache_path(tmp_path / "cache", "d-raw", url, sha) == cache_path(
        tmp_path / "cache", "d-raw", url, recall_das_sha(tmp_path / "cache", "d-raw")
    )


def test_recall_is_none_before_anything_is_cached(tmp_path):
    assert recall_das_sha(tmp_path / "cache", "d-raw") is None


# ------------------------------------------------ recorded Rutgers fixture
# 5000 contiguous rows captured from ru33-20211001T1841-trajectory-raw-delayed
# on 2026-08-24. Real bytes, not a model of what the server sends: it carries
# NaN timestamps, a 0.0 timestamp, and a genuine surface pressure of 0.0.
FIXTURE = REPO_ROOT / "tests/data/erddap_ru33_slice.nc"
_VARS = [
    "sci_ctd41cp_timestamp",
    "sci_water_cond",
    "sci_water_pressure",
    "sci_water_temp",
]


def _fixture() -> xr.Dataset:
    if not FIXTURE.exists():
        pytest.skip("recorded ERDDAP fixture not present")
    return xr.open_dataset(FIXTURE, decode_times=False, mask_and_scale=False)


def test_recorded_slice_sanitises_to_the_measured_counts():
    """Exact tally on real bytes, so a rule change cannot drift unnoticed."""
    from odas_tpw.dinkum.build import resolve_time_bounds, sanitize_time

    ds = _fixture()
    mask_fill_values(ds, _VARS)
    lo, hi = resolve_time_bounds("2021-09-01", "2021-12-01")
    times, stats = sanitize_time(
        np.asarray(ds["sci_ctd41cp_timestamp"].values, dtype=np.float64), lo, hi, "mean"
    )
    assert stats == {
        "n_total": 5000,
        "n_nan": 2724,
        "n_out_of_range": 1,  # the single 0.0 timestamp, below the lower bound
        "n_duplicate": 0,
        "n_kept": 2275,
    }
    assert np.all(np.diff(times) > 0)


def test_fill_masking_on_real_bytes_leaves_the_clock_alone():
    """The CTD's own stamp is NaN where it did not report, never a fill value."""
    ds = _fixture()
    stats = mask_fill_values(ds, _VARS)
    assert stats["sci_ctd41cp_timestamp"]["fill_masked"] == 0
    for name in _VARS[1:]:
        assert stats[name]["fill_masked"] == 2724


def test_a_real_surface_pressure_of_zero_survives():
    """Why `drop_zero_as_fill` is empty by default, on Rutgers' own data.

    Across the full ru33 deployment, 33 rows carry `sci_water_pressure == 0.0`
    on a perfectly good timestamp. They are not sentinels: their neighbours
    read 0.002-0.013 bar, i.e. 2-13 cm of water. The glider is at the surface
    and the sensor is right.

    The Rutgers notebook's blanket `0.0 -> NaN` deletes all 33. We keep them,
    and the row count is identical either way -- which is exactly why this
    needs its own test: the golden row count hides the difference.
    """
    ds = _fixture()
    t = np.asarray(ds["sci_ctd41cp_timestamp"].values, dtype=np.float64)
    good = np.isfinite(t) & (t > 1.6e9)

    mask_fill_values(ds, _VARS)  # default: no zero rule
    p = np.asarray(ds["sci_water_pressure"].values, dtype=np.float64)
    assert np.count_nonzero((p == 0.0) & good) == 1, "the real surface sample was deleted"

    # Opting in removes it -- available, and off for a reason. Two go: the
    # real surface sample, and one on the row whose timestamp is the 0.0
    # sentinel (which the timestamp rule discards anyway, so only the first
    # is a loss).
    ds2 = _fixture()
    stats = mask_fill_values(ds2, _VARS, drop_zero=["sci_water_pressure"])
    assert stats["sci_water_pressure"]["zero_masked"] == 2
    p2 = np.asarray(ds2["sci_water_pressure"].values, dtype=np.float64)
    assert np.count_nonzero((p2 == 0.0) & good) == 0, "the real sample should now be gone"


@pytest.mark.skipif(
    not __import__("os").environ.get("ERDDAP_LIVE"),
    reason="set ERDDAP_LIVE=1 to hit the real server",
)
def test_live_full_deployment_matches_the_notebook():  # pragma: no cover
    """Golden case: the Rutgers notebook's own published answer.

    Its stored outputs record 868,427 rows in and 346,518 out, over
    1633113400.37739 .. 1634738980.72214. We reach the same numbers by a
    different route -- timestamp validity with both bounds, and no blanket
    zero rule -- which is the strongest available evidence that dropping the
    blanket rule costs nothing.
    """
    import urllib.request

    from odas_tpw.dinkum.build import resolve_time_bounds, sanitize_time
    from odas_tpw.erddap.query import tabledap_url

    url = tabledap_url(
        "https://slocum-data.marine.rutgers.edu/erddap",
        "ru33-20211001T1841-trajectory-raw-delayed",
        _VARS,
    )
    path, _ = urllib.request.urlretrieve(url)
    ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False)
    assert ds.sizes["row"] == 868_427
    mask_fill_values(ds, _VARS)
    lo, hi = resolve_time_bounds("2021-09-01", "2021-12-01")
    times, stats = sanitize_time(
        np.asarray(ds["sci_ctd41cp_timestamp"].values, dtype=np.float64), lo, hi, "mean"
    )
    assert stats["n_kept"] == 346_518
    assert times[0] == pytest.approx(1633113400.37739)
    assert times[-1] == pytest.approx(1634738980.72214)
