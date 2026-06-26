import xarray as xr

def harmonize_grids(ds_coarse, ds_fine_target):
    """
    Downscales coarse resolution data to the fine resolution target using bilinear interpolation.
    """
    return ds_coarse.interp(
        lat=ds_fine_target['lat'], 
        lon=ds_fine_target['lon'], 
        method='linear'
    )

def mask_to_region(ds, shapefile_path):
    """
    Clips the xarray dataset to a specific geographical boundary shapefile.
    Imports are lazy to avoid breaking the pipeline if rioxarray/shapely are unavailable.
    """
    import rioxarray
    import geopandas as gpd
    from shapely.geometry import mapping

    gdf = gpd.read_file(shapefile_path)
    ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    ds.rio.write_crs("epsg:4326", inplace=True)
    return ds.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)

def minmax_normalize(data_array):
    """Normalized array bounded between 0 and 1 for stable AI gradients.
    Adds epsilon to denominator to prevent division-by-zero for constant fields."""
    min_val = data_array.min()
    max_val = data_array.max()
    return (data_array - min_val) / (max_val - min_val + 1e-8), min_val, max_val
