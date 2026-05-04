# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""CF/ACDD NetCDF schema definitions.

Python replacement for Matlab's Combo.json and CTD.json.
Provides variable metadata (units, standard_name, long_name) for
CF-1.13 / ACDD-1.3 compliant NetCDF output.

Reference: Code/Combo.json, Code/CTD.json
"""

import xarray as xr

# ---------------------------------------------------------------------------
# UDUNITS fixups for unit strings emitted by RSI's INI config
# ---------------------------------------------------------------------------
#
# RSI .p files carry unit strings in the embedded ASCII config that are
# either UDUNITS-incompatible (``umol_L-1`` instead of ``umol L-1``) or
# ambiguous (``deg`` — degree of arc or degree Celsius?). compliance-checker
# emits hard CF errors on these. The mapping below normalises everything to
# UDUNITS-parseable strings and is applied by :func:`canonicalize_units`
# wherever we write per-channel attributes (e.g. extract_profiles).

UNITS_FIXUP: dict[str, str] = {
    "umol_L-1": "umol L-1",
    "mS_cm-1": "mS cm-1",
    "deg_C": "degree_Celsius",
    "deg": "degree",
    # Turbidity: FTU is widely used in oceanography but absent from UDUNITS.
    # CF guidance is to mark it dimensionless and document in long_name.
    "FTU": "1",
    "NTU": "1",
}


def canonicalize_units(units: str) -> str:
    """Translate an RSI-emitted unit string to a UDUNITS-parseable one."""
    return UNITS_FIXUP.get(units.strip(), units.strip())

# ---------------------------------------------------------------------------
# Schema definitions — maps variable names to CF attributes
# ---------------------------------------------------------------------------

COMBO_SCHEMA: dict[str, dict] = {
    "bin": {
        "units": "m",
        "standard_name": "depth",
        "long_name": "depth bin centre",
        "positive": "down",
        "axis": "Z",
    },
    "profile": {
        "long_name": "profile index",
        "cf_role": "profile_id",
    },
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
        "units_metadata": "temperature: on_scale",
    },
    "T1": {
        "units": "degree_Celsius",
        "standard_name": "sea_water_temperature",
        "long_name": "FP07 thermistor temperature (probe 1)",
        "units_metadata": "temperature: on_scale",
    },
    "T2": {
        "units": "degree_Celsius",
        "standard_name": "sea_water_temperature",
        "long_name": "FP07 thermistor temperature (probe 2)",
        "units_metadata": "temperature: on_scale",
    },
    "DO_T": {
        "units": "degree_Celsius",
        "long_name": "dissolved-oxygen sensor temperature",
        "units_metadata": "temperature: on_scale",
    },
    "DO": {
        "units": "umol L-1",
        "standard_name": "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
        "long_name": "dissolved oxygen",
    },
    "Chlorophyll": {
        "units": "ug L-1",
        "standard_name": "mass_concentration_of_chlorophyll_in_sea_water",
        "long_name": "chlorophyll concentration",
    },
    "Turbidity": {
        "units": "1",
        "long_name": "turbidity (formazin turbidity units)",
    },
    "P_dP": {
        "units": "dbar",
        "long_name": "pressure offset (P - low-pass filtered P)",
    },
    "Gnd": {
        "units": "V",
        "long_name": "ground reference voltage",
    },
    "V_Bat": {
        "units": "V",
        "long_name": "battery voltage",
    },
    "PV": {
        "units": "V",
        "long_name": "vehicle power voltage",
    },
    "Incl_X": {
        "units": "degree",
        "long_name": "inclinometer X angle",
    },
    "Incl_Y": {
        "units": "degree",
        "long_name": "inclinometer Y angle",
    },
    "Incl_T": {
        "units": "degree_Celsius",
        "long_name": "inclinometer temperature",
        "units_metadata": "temperature: on_scale",
    },
    "JAC_T": {
        "nc_name": "temp",
        "units": "degree_Celsius",
        "standard_name": "sea_water_temperature",
        "long_name": "in-situ temperature (JFE)",
        "units_metadata": "temperature: on_scale",
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
        "units_metadata": "temperature: on_scale",
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
        "standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
        "long_name": "TKE dissipation rate (probe 1)",
    },
    "e_2": {
        "nc_name": "e_2",
        "units": "W/kg",
        "standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
        "long_name": "TKE dissipation rate (probe 2)",
    },
    "epsilonMean": {
        "nc_name": "epsilonMean",
        "units": "W/kg",
        "standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
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
    "T_mean": {
        "nc_name": "T_mean",
        "units": "degree_Celsius",
        "long_name": "mean in-situ temperature within FFT window",
        "units_metadata": "temperature: on_scale",
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
    # Shared diagnostic vars carried over from the diss/profile pipeline
    "speed": COMBO_SCHEMA["speed"],
    "nu": COMBO_SCHEMA["nu"],
    "T_mean": COMBO_SCHEMA["T_mean"],
    "lat": COMBO_SCHEMA["lat"],
    "lon": COMBO_SCHEMA["lon"],
    "stime": COMBO_SCHEMA["stime"],
    "etime": COMBO_SCHEMA["etime"],
    "depth": COMBO_SCHEMA["depth"],
    "P": COMBO_SCHEMA["P"],
    "bin": COMBO_SCHEMA["bin"],
    "profile": COMBO_SCHEMA["profile"],
}

CTD_SCHEMA: dict[str, dict] = {
    "time": {
        "nc_name": "time",
        "units": "seconds since 1970-01-01",
        "standard_name": "time",
        "long_name": "time bin center",
    },
    "JAC_T": COMBO_SCHEMA["JAC_T"],
    "JAC_C": COMBO_SCHEMA["JAC_C"],
    "P": COMBO_SCHEMA["P"],
    "depth": COMBO_SCHEMA["depth"],
    "SP": COMBO_SCHEMA["SP"],
    "SA": COMBO_SCHEMA["SA"],
    "CT": COMBO_SCHEMA["CT"],
    "sigma0": COMBO_SCHEMA["sigma0"],
    "rho": COMBO_SCHEMA["rho"],
    "lat": COMBO_SCHEMA["lat"],
    "lon": COMBO_SCHEMA["lon"],
    "DO": COMBO_SCHEMA["DO"],
    "Chlorophyll": COMBO_SCHEMA["Chlorophyll"],
    "Turbidity": COMBO_SCHEMA["Turbidity"],
}

# Fixed global attributes for CF-1.13 / ACDD-1.3 compliance
GLOBAL_ATTRS: dict[str, str] = {
    "Conventions": "CF-1.13, ACDD-1.3",
    "standard_name_vocabulary": "CF Standard Name Table v89",
    "featureType": "profile",
}


_COVERAGE_CONTENT_TYPE: dict[str, str] = {
    # Geometric / time coordinate variables
    "bin": "coordinate",
    "depth": "coordinate",
    "profile": "coordinate",
    "time": "coordinate",
    "stime": "coordinate",
    "etime": "coordinate",
    "lat": "coordinate",
    "lon": "coordinate",
    # Auxiliary diagnostics — CF/ACDD distinguishes these from the actual
    # measured field.  ACDD §coverage_content_type calls this
    # ``auxiliaryInformation``.
    "FM_1": "auxiliaryInformation",
    "FM_2": "auxiliaryInformation",
    "fom_T_1": "auxiliaryInformation",
    "fom_T_2": "auxiliaryInformation",
    "K_max_1": "auxiliaryInformation",
    "K_max_2": "auxiliaryInformation",
    "K_max_T_1": "auxiliaryInformation",
    "K_max_T_2": "auxiliaryInformation",
    "epsilonLnSigma": "auxiliaryInformation",
    "n_samples": "auxiliaryInformation",
    "Gnd": "auxiliaryInformation",
    "PV": "auxiliaryInformation",
    "P_dP": "auxiliaryInformation",
    "V_Bat": "auxiliaryInformation",
    "Incl_X": "auxiliaryInformation",
    "Incl_Y": "auxiliaryInformation",
    "Incl_T": "auxiliaryInformation",
    "DO_T": "auxiliaryInformation",
    "speed": "auxiliaryInformation",
    "nu": "auxiliaryInformation",
}


def coverage_content_type_for(var_name: str) -> str:
    """Return ACDD ``coverage_content_type`` for *var_name*.

    Defaults to ``physicalMeasurement`` for any variable not explicitly
    classified — that is the right call for T, S, P, epsilon, chi, density,
    DO, etc.  Coordinate vars and pure diagnostics get their own values.
    """
    return _COVERAGE_CONTENT_TYPE.get(var_name, "physicalMeasurement")


def apply_schema(ds: xr.Dataset, schema: dict[str, dict]) -> xr.Dataset:
    """Apply CF attributes from *schema* to variables in *ds*.

    For each variable in *ds* that has a matching entry in *schema*,
    the schema's metadata (units, standard_name, long_name, etc.) is
    applied as variable attributes.  Every variable also receives an ACDD
    ``coverage_content_type`` (auxiliary diagnostics, coordinates, and
    physical measurements all get distinct types).

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
        ds[vname].attrs.setdefault("coverage_content_type", coverage_content_type_for(vname))
    return ds
