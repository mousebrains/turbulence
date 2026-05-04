# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.gps — GPS providers."""

import numpy as np
import pytest

from odas_tpw.perturb.gps import (
    GPSFixed,
    GPSFromCSV,
    GPSFromNetCDF,
    GPSNaN,
    GPSProvider,
    create_gps,
)


class TestGPSNaN:
    def test_returns_nan(self):
        gps = GPSNaN()
        t = np.array([0.0, 1.0, 2.0])
        assert np.all(np.isnan(gps.lat(t)))
        assert np.all(np.isnan(gps.lon(t)))

    def test_correct_shape(self):
        gps = GPSNaN()
        t = np.arange(10.0)
        assert gps.lat(t).shape == (10,)

    def test_is_provider(self):
        assert isinstance(GPSNaN(), GPSProvider)


class TestGPSFixed:
    def test_returns_constants(self):
        gps = GPSFixed(15.0, 145.0)
        t = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(gps.lat(t), 15.0)
        np.testing.assert_allclose(gps.lon(t), 145.0)

    def test_correct_shape(self):
        gps = GPSFixed(10.0, 20.0)
        t = np.arange(5.0)
        assert gps.lat(t).shape == (5,)

    def test_is_provider(self):
        assert isinstance(GPSFixed(0, 0), GPSProvider)


class TestCreateGPS:
    def test_nan_source(self):
        gps = create_gps({"source": "nan"})
        assert isinstance(gps, GPSNaN)

    def test_fixed_source(self):
        gps = create_gps({"source": "fixed", "lat": 15.0, "lon": 145.0})
        assert isinstance(gps, GPSFixed)
        t = np.array([0.0])
        assert gps.lat(t)[0] == 15.0
        assert gps.lon(t)[0] == 145.0

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown GPS source"):
            create_gps({"source": "unknown"})

    def test_csv_source(self, tmp_path):
        import pandas as pd

        csv_file = tmp_path / "gps.csv"
        pd.DataFrame({"t": [0, 1, 2], "lat": [10, 20, 30], "lon": [100, 110, 120]}).to_csv(
            csv_file,
            index=False,
        )
        gps = create_gps({"source": "csv", "file": str(csv_file)})
        assert isinstance(gps, GPSFromCSV)
        t = np.array([1.0])
        np.testing.assert_allclose(gps.lat(t), 20.0)
        np.testing.assert_allclose(gps.lon(t), 110.0)

    def test_netcdf_source(self, tmp_path):
        import netCDF4 as nc

        nc_file = tmp_path / "gps.nc"
        ds = nc.Dataset(str(nc_file), "w")
        ds.createDimension("obs", 3)
        t_var = ds.createVariable("time", "f8", ("obs",))
        lat_var = ds.createVariable("lat", "f8", ("obs",))
        lon_var = ds.createVariable("lon", "f8", ("obs",))
        t_var[:] = [0, 1, 2]
        lat_var[:] = [10, 20, 30]
        lon_var[:] = [100, 110, 120]
        ds.close()

        gps = create_gps({"source": "netcdf", "file": str(nc_file)})
        assert isinstance(gps, GPSFromNetCDF)
        t = np.array([1.0])
        np.testing.assert_allclose(gps.lat(t), 20.0)
        np.testing.assert_allclose(gps.lon(t), 110.0)

    def test_default_is_nan(self):
        gps = create_gps({})
        assert isinstance(gps, GPSNaN)


class TestGPSFromCSV:
    def _make_csv(self, path, time_col="t", lat_col="lat", lon_col="lon"):
        import pandas as pd

        df = pd.DataFrame(
            {time_col: [0, 1, 2], lat_col: [10, 20, 30], lon_col: [100, 110, 120]},
        )
        df.to_csv(path, index=False)
        return path

    def test_interpolation(self, tmp_path):
        csv_file = self._make_csv(tmp_path / "gps.csv")
        gps = GPSFromCSV(csv_file)
        t = np.array([0.5])
        np.testing.assert_allclose(gps.lat(t), 15.0)
        np.testing.assert_allclose(gps.lon(t), 105.0)

    def test_extrapolation(self, tmp_path):
        csv_file = self._make_csv(tmp_path / "gps.csv")
        gps = GPSFromCSV(csv_file)
        t = np.array([-1.0, 5.0])
        lat = gps.lat(t)
        lon = gps.lon(t)
        # Extrapolated values should not be NaN
        assert not np.any(np.isnan(lat))
        assert not np.any(np.isnan(lon))
        # Linear extrapolation: slope is 10/1 for lat, 10/1 for lon
        np.testing.assert_allclose(lat[0], 0.0)  # 10 + (-1)*10 = 0
        np.testing.assert_allclose(lat[1], 60.0)  # 10 + 5*10 = 60
        np.testing.assert_allclose(lon[0], 90.0)  # 100 + (-1)*10 = 90
        np.testing.assert_allclose(lon[1], 150.0)  # 100 + 5*10 = 150

    def test_custom_columns(self, tmp_path):
        csv_file = self._make_csv(
            tmp_path / "gps.csv",
            time_col="epoch",
            lat_col="latitude",
            lon_col="longitude",
        )
        gps = GPSFromCSV(csv_file, time_col="epoch", lat_col="latitude", lon_col="longitude")
        t = np.array([0.5])
        np.testing.assert_allclose(gps.lat(t), 15.0)
        np.testing.assert_allclose(gps.lon(t), 105.0)

    def test_is_provider(self, tmp_path):
        csv_file = self._make_csv(tmp_path / "gps.csv")
        assert isinstance(GPSFromCSV(csv_file), GPSProvider)

    def test_datetime64_column_normalises_via_helper(self):
        """``_to_epoch_seconds`` collapses datetime64 columns to float
        epoch seconds.  This is the path that `GPSFromCSV` takes when the
        caller hands it a pandas-parsed datetime column."""
        from odas_tpw.perturb.gps import _to_epoch_seconds

        # 2023-01-01T00:00:00 UTC, +1 s, +2 s — datetime64[ns] like
        # pd.read_csv(parse_dates=...) would yield.
        arr = np.array(
            [
                np.datetime64("2023-01-01T00:00:00", "ns"),
                np.datetime64("2023-01-01T00:00:01", "ns"),
                np.datetime64("2023-01-01T00:00:02", "ns"),
            ]
        )
        out = _to_epoch_seconds(arr)
        assert out.dtype == np.float64
        # 2023-01-01T00:00:00Z is 1672531200 epoch seconds.
        np.testing.assert_allclose(out, [1672531200.0, 1672531201.0, 1672531202.0])

    def test_helper_passthrough_for_numeric(self):
        """Numeric columns are passed through (callers are responsible for
        ensuring epoch seconds)."""
        from odas_tpw.perturb.gps import _to_epoch_seconds

        out = _to_epoch_seconds(np.array([1.0, 2.0, 3.0]))
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


class TestGPSFromNetCDF:
    def _make_nc(self, path, time_var="time", lat_var="lat", lon_var="lon"):
        import netCDF4 as nc

        ds = nc.Dataset(str(path), "w")
        ds.createDimension("obs", 3)
        t = ds.createVariable(time_var, "f8", ("obs",))
        lat = ds.createVariable(lat_var, "f8", ("obs",))
        lon = ds.createVariable(lon_var, "f8", ("obs",))
        t[:] = [0, 1, 2]
        lat[:] = [10, 20, 30]
        lon[:] = [100, 110, 120]
        ds.close()
        return path

    def test_interpolation(self, tmp_path):
        nc_file = self._make_nc(tmp_path / "gps.nc")
        gps = GPSFromNetCDF(nc_file)
        t = np.array([0.5])
        np.testing.assert_allclose(gps.lat(t), 15.0)
        np.testing.assert_allclose(gps.lon(t), 105.0)

    def test_extrapolation(self, tmp_path):
        nc_file = self._make_nc(tmp_path / "gps.nc")
        gps = GPSFromNetCDF(nc_file)
        t = np.array([-1.0, 5.0])
        lat = gps.lat(t)
        lon = gps.lon(t)
        assert not np.any(np.isnan(lat))
        assert not np.any(np.isnan(lon))
        np.testing.assert_allclose(lat[0], 0.0)
        np.testing.assert_allclose(lat[1], 60.0)

    def test_custom_variables(self, tmp_path):
        nc_file = self._make_nc(
            tmp_path / "gps.nc",
            time_var="epoch",
            lat_var="latitude",
            lon_var="longitude",
        )
        gps = GPSFromNetCDF(nc_file, time_var="epoch", lat_var="latitude", lon_var="longitude")
        t = np.array([1.5])
        np.testing.assert_allclose(gps.lat(t), 25.0)
        np.testing.assert_allclose(gps.lon(t), 115.0)

    def test_is_provider(self, tmp_path):
        nc_file = self._make_nc(tmp_path / "gps.nc")
        assert isinstance(GPSFromNetCDF(nc_file), GPSProvider)

    def test_cf_time_units_decoded_to_epoch_seconds(self, tmp_path):
        """A NetCDF time var with CF units (e.g. 'milliseconds since ...')
        is decoded to epoch seconds, so callers query with epoch-second
        offsets. Mirrors the ARCTERX gps.nc layout.
        """
        import netCDF4 as nc

        path = tmp_path / "gps_cf.nc"
        ds = nc.Dataset(str(path), "w")
        ds.createDimension("t", 3)
        tvar = ds.createVariable("t", "i8", ("t",))
        tvar.units = "milliseconds since 2023-05-03 06:51:31"
        tvar.calendar = "proleptic_gregorian"
        tvar[:] = [0, 1000, 2000]  # 0/1/2 seconds after the reference
        lat = ds.createVariable("lat", "f8", ("t",))
        lon = ds.createVariable("lon", "f8", ("t",))
        lat[:] = [7.0, 8.0, 9.0]
        lon[:] = [134.0, 135.0, 136.0]
        ds.close()

        gps = GPSFromNetCDF(str(path), time_var="t")
        # Reference epoch seconds for 2023-05-03 06:51:31 UTC.
        from datetime import UTC, datetime

        t0 = datetime(2023, 5, 3, 6, 51, 31, tzinfo=UTC).timestamp()
        np.testing.assert_allclose(gps.lat(np.array([t0])), [7.0])
        np.testing.assert_allclose(gps.lat(np.array([t0 + 1.0])), [8.0])
        np.testing.assert_allclose(gps.lon(np.array([t0 + 2.0])), [136.0])
