# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""CF/ACDD NetCDF schema definitions.

Python replacement for Matlab's Combo.json and CTD.json.
Provides variable metadata (units, standard_name, long_name) for
CF-1.8 / ACDD-1.3 compliant NetCDF output.

Reference: Code/Combo.json, Code/CTD.json
"""

import xarray as xr

# ---------------------------------------------------------------------------
# Schema definitions — maps variable names to CF attributes
# ---------------------------------------------------------------------------

COMBO_SCHEMA: dict[str, dict] = {
    "time": {
        "nc_name": "time",
        "units": "seconds since 1970-01-01",
        "standard_name": "time",
        "long_name": "time",
    },
    "stime": {
        "nc_name": "stime",
        "units": "seconds since 1970-01-01",
        "standard_name": "time",
        "long_name": "profile start time",
    },
    "etime": {
        "nc_name": "etime",
        "units": "seconds since 1970-01-01",
        "standard_name": "time",
        "long_name": "profile end time",
    },
    "depth": {
        "nc_name": "depth",
        "units": "m",
        "standard_name": "depth",
        "long_name": "depth",
        "positive": "down",
    },
    "P": {
        "nc_name": "pressure",
        "units": "dbar",
        "standard_name": "sea_water_pressure",
        "long_name": "sea water pressure",
    },
    "T": {
        "nc_name": "temp",
        "units": "degree_Celsius",
        "standard_name": "sea_water_temperature",
        "long_name": "in-situ temperature",
    },
    "JAC_T": {
        "nc_name": "temp",
        "units": "degree_Celsius",
        "standard_name": "sea_water_temperature",
        "long_name": "in-situ temperature (JFE)",
    },
    "JAC_C": {
        "nc_name": "cond",
        "units": "mS/cm",
        "standard_name": "sea_water_electrical_conductivity",
        "long_name": "conductivity (JFE)",
    },
    "SP": {
        "nc_name": "SP",
        "units": "PSU",
        "standard_name": "sea_water_practical_salinity",
        "long_name": "practical salinity",
    },
    "SA": {
        "nc_name": "SA",
        "units": "g/kg",
        "standard_name": "sea_water_absolute_salinity",
        "long_name": "absolute salinity",
    },
    "CT": {
        "nc_name": "CT",
        "units": "degree_Celsius",
        "standard_name": "sea_water_conservative_temperature",
        "long_name": "conservative temperature",
    },
    "sigma0": {
        "nc_name": "sigma0",
        "units": "kg/m^3",
        "standard_name": "sea_water_sigma_theta",
        "long_name": "potential density anomaly",
    },
    "rho": {
        "nc_name": "rho",
        "units": "kg/m^3",
        "standard_name": "sea_water_density",
        "long_name": "in-situ density - 1000",
    },
    "lat": {
        "nc_name": "lat",
        "units": "degrees_north",
        "standard_name": "latitude",
        "long_name": "latitude",
    },
    "lon": {
        "nc_name": "lon",
        "units": "degrees_east",
        "standard_name": "longitude",
        "long_name": "longitude",
    },
    "e_1": {
        "nc_name": "e_1",
        "units": "W/kg",
        "long_name": "TKE dissipation rate (probe 1)",
    },
    "e_2": {
        "nc_name": "e_2",
        "units": "W/kg",
        "long_name": "TKE dissipation rate (probe 2)",
    },
    "epsilonMean": {
        "nc_name": "epsilonMean",
        "units": "W/kg",
        "long_name": "combined TKE dissipation rate",
    },
    "epsilonLnSigma": {
        "nc_name": "epsilonLnSigma",
        "units": "",
        "long_name": "sigma of ln(epsilon)",
    },
    "FM_1": {
        "nc_name": "FM_1",
        "units": "",
        "long_name": "figure of merit (probe 1)",
    },
    "FM_2": {
        "nc_name": "FM_2",
        "units": "",
        "long_name": "figure of merit (probe 2)",
    },
    "K_max_1": {
        "nc_name": "K_max_1",
        "units": "cpm",
        "long_name": "maximum resolved wavenumber (probe 1)",
    },
    "K_max_2": {
        "nc_name": "K_max_2",
        "units": "cpm",
        "long_name": "maximum resolved wavenumber (probe 2)",
    },
    "speed": {
        "nc_name": "speed",
        "units": "m/s",
        "long_name": "profiling speed",
    },
    "nu": {
        "nc_name": "nu",
        "units": "m^2/s",
        "long_name": "kinematic viscosity",
    },
}

CHI_SCHEMA: dict[str, dict] = {
    "chi_1": {
        "nc_name": "chi_1",
        "units": "K^2/s",
        "long_name": "thermal variance dissipation (probe 1)",
    },
    "chi_2": {
        "nc_name": "chi_2",
        "units": "K^2/s",
        "long_name": "thermal variance dissipation (probe 2)",
    },
    "kB_1": {
        "nc_name": "kB_1",
        "units": "cpm",
        "long_name": "Batchelor wavenumber (probe 1)",
    },
    "kB_2": {
        "nc_name": "kB_2",
        "units": "cpm",
        "long_name": "Batchelor wavenumber (probe 2)",
    },
    "fom_T_1": {
        "nc_name": "fom_T_1",
        "units": "",
        "long_name": "figure of merit chi (probe 1)",
    },
    "fom_T_2": {
        "nc_name": "fom_T_2",
        "units": "",
        "long_name": "figure of merit chi (probe 2)",
    },
    "K_max_T_1": {
        "nc_name": "K_max_T_1",
        "units": "cpm",
        "long_name": "max wavenumber chi (probe 1)",
    },
    "K_max_T_2": {
        "nc_name": "K_max_T_2",
        "units": "cpm",
        "long_name": "max wavenumber chi (probe 2)",
    },
}

CTD_SCHEMA: dict[str, dict] = {
    "time": {
        "nc_name": "time",
        "units": "seconds since 1970-01-01",
        "standard_name": "time",
        "long_name": "time bin center",
    },
    "JAC_T": {
        "nc_name": "temp",
        "units": "degree_Celsius",
        "standard_name": "sea_water_temperature",
        "long_name": "in-situ temperature",
    },
    "JAC_C": {
        "nc_name": "cond",
        "units": "mS/cm",
        "standard_name": "sea_water_electrical_conductivity",
        "long_name": "conductivity",
    },
    "P": {
        "nc_name": "pressure",
        "units": "dbar",
        "standard_name": "sea_water_pressure",
        "long_name": "sea water pressure",
    },
    "depth": {
        "nc_name": "depth",
        "units": "m",
        "standard_name": "depth",
        "long_name": "depth",
        "positive": "down",
    },
    "SP": COMBO_SCHEMA["SP"],
    "SA": COMBO_SCHEMA["SA"],
    "CT": COMBO_SCHEMA["CT"],
    "sigma0": COMBO_SCHEMA["sigma0"],
    "rho": COMBO_SCHEMA["rho"],
    "lat": COMBO_SCHEMA["lat"],
    "lon": COMBO_SCHEMA["lon"],
    "DO": {
        "nc_name": "DO",
        "units": "umol/kg",
        "standard_name": "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
        "long_name": "dissolved oxygen",
    },
    "Chlorophyll": {
        "nc_name": "chlorophyll",
        "units": "ug/L",
        "standard_name": "mass_concentration_of_chlorophyll_in_sea_water",
        "long_name": "chlorophyll concentration",
    },
    "Turbidity": {
        "nc_name": "turbidity",
        "units": "NTU",
        "standard_name": "sea_water_turbidity",
        "long_name": "turbidity",
    },
}

# Fixed global attributes for CF-1.8 / ACDD-1.3 compliance
GLOBAL_ATTRS: dict[str, str] = {
    "Conventions": "CF-1.8, ACDD-1.3",
    "standard_name_vocabulary": "CF Standard Name Table v83",
    "featureType": "profile",
}


def apply_schema(ds: xr.Dataset, schema: dict[str, dict]) -> xr.Dataset:
    """Apply CF attributes from *schema* to variables in *ds*.

    For each variable in *ds* that has a matching entry in *schema*,
    the schema's metadata (units, standard_name, long_name, etc.) is
    applied as variable attributes.

    Parameters
    ----------
    ds : xr.Dataset
    schema : dict mapping variable names to attribute dicts.

    Returns
    -------
    xr.Dataset with updated attributes.
    """
    ds = ds.copy()
    for vname in [str(v) for v in ds.data_vars] + [str(c) for c in ds.coords]:
        if vname in schema:
            attrs = schema[vname]
            for key, val in attrs.items():
                if key != "nc_name":
                    ds[vname].attrs[key] = val
    return ds
