import numpy as np
import xarray as xr
import pandas as pd

def parse_imd_rainfall_025(file_path, year, is_leap_year=False):
    """
    Parses IMD 0.25 x 0.25 degree daily gridded rainfall binary files.
    Grid specs: 135 (lon) x 129 (lat) bounded within [66.5E-100.0E, 6.5N-38.5N]
    """
    nlon, nlat = 135, 129
    lon_vals = np.arange(66.5, 66.5 + (nlon * 0.25), 0.25)
    lat_vals = np.arange(6.5, 6.5 + (nlat * 0.25), 0.25)
    days = 366 if is_leap_year else 365
    
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype='<f4')
    
    # Reshape matching the FORTRAN grid layout provided by IMD
    reshaped = raw_data.reshape(days, nlat, nlon)
    reshaped[reshaped == -999.0] = np.nan  # Mask missing data
    
    time_idx = pd.date_range(start=f'{year}-01-01', periods=days, freq='D')
    
    return xr.Dataset(
        {"rainfall": (("time", "lat", "lon"), reshaped)},
        coords={"time": time_idx, "lat": lat_vals, "lon": lon_vals}
    )

def parse_imd_temp_100(file_path, year, is_leap_year=False):
    """
    Parses IMD 1.0 x 1.0 degree daily gridded temperature binary files.
    Grid specs: 31 (lon) x 31 (lat) bounded within [67.5E-97.5E, 7.5N-37.5N]
    """
    nlon, nlat = 31, 31
    lon_vals = np.arange(67.5, 67.5 + (nlon * 1.0), 1.0)
    lat_vals = np.arange(7.5, 7.5 + (nlat * 1.0), 1.0)
    days = 366 if is_leap_year else 365
    
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype='<f4')
    
    reshaped = raw_data.reshape(days, nlat, nlon)
    reshaped[reshaped == 99.9] = np.nan  # Mask missing data
    
    time_idx = pd.date_range(start=f'{year}-01-01', periods=days, freq='D')
    
    return xr.Dataset(
        {"temperature": (("time", "lat", "lon"), reshaped)},
        coords={"time": time_idx, "lat": lat_vals, "lon": lon_vals}
    )
