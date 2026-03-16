# Tests for odas_tpw.rsi.vehicle
"""Unit tests for vehicle attribute resolution."""

from odas_tpw.rsi.vehicle import resolve_direction, resolve_tau


class TestResolveDirection:
    """resolve_direction() maps (direction, vehicle) → resolved direction."""

    def test_auto_vmp_gives_down(self):
        assert resolve_direction("auto", "vmp") == "down"

    def test_auto_rvmp_gives_up(self):
        assert resolve_direction("auto", "rvmp") == "up"

    def test_auto_slocum_gives_glide(self):
        assert resolve_direction("auto", "slocum_glider") == "glide"

    def test_auto_sea_glider_gives_glide(self):
        assert resolve_direction("auto", "sea_glider") == "glide"

    def test_auto_auv_gives_horizontal(self):
        assert resolve_direction("auto", "auv") == "horizontal"

    def test_auto_unknown_gives_default_down(self):
        assert resolve_direction("auto", "unknown_vehicle") == "down"

    def test_auto_empty_gives_default_down(self):
        assert resolve_direction("auto", "") == "down"

    def test_explicit_up_overrides_vmp(self):
        """Explicit direction overrides vehicle default."""
        assert resolve_direction("up", "vmp") == "up"

    def test_explicit_down_overrides_rvmp(self):
        assert resolve_direction("down", "rvmp") == "down"

    def test_explicit_glide_passes_through(self):
        assert resolve_direction("glide", "vmp") == "glide"

    def test_explicit_horizontal_passes_through(self):
        assert resolve_direction("horizontal", "vmp") == "horizontal"

    def test_case_insensitive_vehicle(self):
        assert resolve_direction("auto", "Slocum_Glider") == "glide"

    def test_case_insensitive_direction(self):
        assert resolve_direction("UP", "vmp") == "up"


class TestResolveTau:
    """resolve_tau() returns the smoothing time constant for a vehicle."""

    def test_vmp_tau(self):
        assert resolve_tau("vmp") == 1.5

    def test_slocum_tau(self):
        assert resolve_tau("slocum_glider") == 3.0

    def test_sea_glider_tau(self):
        assert resolve_tau("sea_glider") == 5.0

    def test_argo_float_tau(self):
        assert resolve_tau("argo_float") == 60.0

    def test_unknown_gives_default(self):
        assert resolve_tau("unknown_vehicle") == 1.5

    def test_empty_gives_default(self):
        assert resolve_tau("") == 1.5

    def test_case_insensitive(self):
        assert resolve_tau("Slocum_Glider") == 3.0
