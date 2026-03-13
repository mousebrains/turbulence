# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.netcdf_schema — CF/ACDD schema definitions."""

import xarray as xr

from perturb.netcdf_schema import (
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

    def test_all_entries_have_units(self):
        schemas = [("COMBO", COMBO_SCHEMA), ("CHI", CHI_SCHEMA), ("CTD", CTD_SCHEMA)]
        for schema_name, schema in schemas:
            for var, attrs in schema.items():
                assert "units" in attrs, f"{schema_name}.{var} missing units"

    def test_all_entries_have_long_name(self):
        schemas = [("COMBO", COMBO_SCHEMA), ("CHI", CHI_SCHEMA), ("CTD", CTD_SCHEMA)]
        for schema_name, schema in schemas:
            for var, attrs in schema.items():
                assert "long_name" in attrs, f"{schema_name}.{var} missing long_name"

    def test_global_attrs_cf_convention(self):
        assert "Conventions" in GLOBAL_ATTRS
        assert "CF-1.8" in GLOBAL_ATTRS["Conventions"]


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
