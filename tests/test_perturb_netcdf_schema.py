# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.netcdf_schema — CF/ACDD schema definitions."""

import xarray as xr

from odas_tpw.perturb.netcdf_schema import (
    CHI_SCHEMA,
    COMBO_SCHEMA,
    CTD_SCHEMA,
    GLOBAL_ATTRS,
    apply_schema,
)


class TestSchemas:
    def test_combo_has_required_vars(self):
        for key in ["depth", "lat", "lon", "e_1", "e_2", "epsilonMean", "speed"]:
            assert key in COMBO_SCHEMA

    def test_chi_has_required_vars(self):
        for key in ["chi_1", "chi_2", "kB_1", "kB_2"]:
            assert key in CHI_SCHEMA

    def test_ctd_has_required_vars(self):
        for key in ["JAC_T", "JAC_C", "P", "depth", "SP", "SA", "lat", "lon"]:
            assert key in CTD_SCHEMA

    def test_practical_salinity_units_are_cf_dimensionless(self):
        """Practical salinity must be CF/TEOS-10 dimensionless "1", not the
        non-UDUNITS "PSU"; the standard_name carries the meaning (#38)."""
        sp = CTD_SCHEMA["SP"]
        assert sp["units"] == "1"
        assert sp["standard_name"] == "sea_water_practical_salinity"
        assert "PSU" in sp["long_name"]  # PSU retained in the human-readable name

    def test_all_entries_have_units(self):
        # ``profile`` is a CF profile-id index (cf_role) — has no physical
        # units and CF-1.13 §9.5 specifies it deliberately omits ``units``.
        skip_unitless = {"profile"}
        schemas = [("COMBO", COMBO_SCHEMA), ("CHI", CHI_SCHEMA), ("CTD", CTD_SCHEMA)]
        for schema_name, schema in schemas:
            for var, attrs in schema.items():
                if var in skip_unitless:
                    continue
                assert "units" in attrs, f"{schema_name}.{var} missing units"

    def test_all_entries_have_long_name(self):
        schemas = [("COMBO", COMBO_SCHEMA), ("CHI", CHI_SCHEMA), ("CTD", CTD_SCHEMA)]
        for schema_name, schema in schemas:
            for var, attrs in schema.items():
                assert "long_name" in attrs, f"{schema_name}.{var} missing long_name"

    def test_global_attrs_cf_convention(self):
        assert "Conventions" in GLOBAL_ATTRS
        assert "CF-1.13" in GLOBAL_ATTRS["Conventions"]
        assert "ACDD-1.3" in GLOBAL_ATTRS["Conventions"]


class TestChannelMetadata:
    def test_pv_is_pressure_transducer_voltage(self):
        """PV is the pressure transducer's voltage, not vehicle power."""
        assert COMBO_SCHEMA["PV"]["long_name"] == "pressure transducer voltage"

    def test_w_slow_carries_units_and_long_name(self):
        """W_slow (profiling rate) needs self-describing metadata; it is a
        computed non-schema channel otherwise stripped of units by binning."""
        assert COMBO_SCHEMA["W_slow"]["units"] == "dbar s-1"
        assert "long_name" in COMBO_SCHEMA["W_slow"]

    def test_nu_units_are_cf_canonical(self):
        """Kinematic viscosity uses CF-canonical "m2 s-1", not the "m^2/s"
        caret form (matching the sigma0/rho kg m-3 convention)."""
        assert COMBO_SCHEMA["nu"]["units"] == "m2 s-1"

    def test_chi_units_are_cf_canonical(self):
        """Thermal-variance dissipation uses CF-canonical "K2 s-1", not the
        "K^2/s" caret form (matching the RSI-side chi output)."""
        for v in ("chiMean", "chi_1", "chi_2"):
            assert CHI_SCHEMA[v]["units"] == "K2 s-1"


class TestApplySchema:
    def test_applies_attributes(self):
        import numpy as np

        ds = xr.Dataset({"depth": (["time"], np.array([1.0, 2.0, 3.0]))})
        ds = apply_schema(ds, COMBO_SCHEMA)
        assert ds["depth"].attrs["units"] == "m"
        assert ds["depth"].attrs["standard_name"] == "depth"

    def test_ignores_unknown_vars(self):
        import numpy as np

        ds = xr.Dataset({"unknown_var": (["time"], np.array([1.0]))})
        ds = apply_schema(ds, COMBO_SCHEMA)
        assert "units" not in ds["unknown_var"].attrs

    def test_does_not_modify_original(self):
        import numpy as np

        ds = xr.Dataset({"depth": (["time"], np.array([1.0]))})
        apply_schema(ds, COMBO_SCHEMA)
        # Original should not be modified (we copy internally)
        assert "units" not in ds["depth"].attrs
