import os
import json
import xarray as xr
from pathlib import Path

# Import our custom logic
from data_parser import parse_imd_rainfall_025, parse_imd_temp_100
from preprocessing import harmonize_grids, minmax_normalize

def check_drive_mount():
    """Asserts that Google Drive is mounted for Colab execution."""
    if os.path.exists('/content'):
        if not os.path.exists('/content/drive'):
            print("WARNING: Google Drive is not mounted at /content/drive. Saving locally.")
            return False
        return True
    return False

def load_real_data(grd_folder_path, year=2023):
    """
    Ingests raw IMD .grd files, harmonizes their resolutions, normalizes values,
    and returns a merged xarray dataset ready for PyTorch.
    """
    base_dir = Path(grd_folder_path)
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    # 1. Paths based on the strict naming convention from download_sample.py
    rf_path = base_dir / f"ind{year}_rfp25.grd"
    maxt_path = base_dir / f"MaxT_{year}.GRD"
    mint_path = base_dir / f"MinT_{year}.GRD"
    
    # Check existence
    for p in [rf_path, maxt_path, mint_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing raw data file: {p}")
    
    print("Ingesting raw IMD binary files...")
    ds_rain = parse_imd_rainfall_025(rf_path, year, is_leap_year=is_leap)
    ds_maxt = parse_imd_temp_100(maxt_path, year, is_leap_year=is_leap).rename({'temperature': 'max_temp'})
    ds_mint = parse_imd_temp_100(mint_path, year, is_leap_year=is_leap).rename({'temperature': 'min_temp'})
    
    print("Harmonizing spatial grids (Bilinear Interpolation)...")
    # Upsample the 1.0 degree temp grids to the 0.25 degree rainfall grid
    ds_maxt_harmonized = harmonize_grids(ds_maxt, ds_rain)
    ds_mint_harmonized = harmonize_grids(ds_mint, ds_rain)
    
    # Merge into a single xarray Dataset
    ds_merged = xr.merge([ds_rain, ds_maxt_harmonized, ds_mint_harmonized])
    
    print("Applying Min-Max Normalization...")
    config_dict = {}
    
    # Normalize each channel and record bounds
    for var in list(ds_merged.data_vars):
        normalized_da, min_v, max_v = minmax_normalize(ds_merged[var])
        ds_merged[var] = normalized_da
        
        # Convert scalar xarray dataarrays to standard python floats for JSON
        config_dict[var] = {
            "min": float(min_v.values),
            "max": float(max_v.values)
        }
        
    # Determine config save path
    has_drive = check_drive_mount()
    if has_drive:
        save_dir = Path('/content/drive/MyDrive/ISRO_Hackathon_PS5/models')
    else:
        save_dir = Path('./models')
        
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / 'normalization_config.json'
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    print(f"Normalization configuration saved to {config_path}")
    
    return ds_merged
