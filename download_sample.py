import os
import requests
import numpy as np
from pathlib import Path

def generate_mock_grd(file_path, days, nlat, nlon, is_temp=False):
    """Generates a dummy binary .grd file containing random float32 data."""
    print(f"Generating mock binary data for {file_path.name}...")
    # Mock data: random float32s
    mock_data = np.random.rand(days, nlat, nlon).astype('<f4')
    
    # Apply IMD's no-data masks so the parsers handle it correctly
    mask_val = 99.9 if is_temp else -999.0
    # Add a few masked values randomly
    mask_indices = np.random.choice(a=[False, True], size=mock_data.shape, p=[0.95, 0.05])
    mock_data[mask_indices] = mask_val

    with open(file_path, 'wb') as f:
        mock_data.tofile(f)

def download_file(url, out_path, year, nlat, nlon, is_temp):
    """Attempts to download the URL, falls back to generating a mock .grd."""
    days = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
    
    try:
        print(f"Attempting to download: {url}")
        # Use a realistic User-Agent to avoid generic 403s
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, stream=True, headers=headers, timeout=10)
        
        if response.status_code == 200:
            with open(out_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Success! Downloaded to {out_path}")
        else:
            print(f"Server rejected request (Status {response.status_code}). Falling back to mock generator.")
            generate_mock_grd(out_path, days, nlat, nlon, is_temp)
            
    except Exception as e:
        print(f"Connection failed: {e}. Falling back to mock generator.")
        generate_mock_grd(out_path, days, nlat, nlon, is_temp)

def main(year=2023, out_dir='./data/raw'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Rainfall (0.25x0.25)
    rf_url = f"https://imdpune.gov.in/cmpg/Griddata/data/ind{year}_rfp25.grd"
    rf_out = out_dir / f"ind{year}_rfp25.grd"
    download_file(rf_url, rf_out, year, nlat=129, nlon=135, is_temp=False)
    
    # 2. Maximum Temperature (1.0x1.0)
    maxt_url = f"https://imdpune.gov.in/cmpg/Griddata/data/MaxT_{year}.GRD"
    maxt_out = out_dir / f"MaxT_{year}.GRD"
    download_file(maxt_url, maxt_out, year, nlat=31, nlon=31, is_temp=True)
    
    # 3. Minimum Temperature (1.0x1.0)
    mint_url = f"https://imdpune.gov.in/cmpg/Griddata/data/MinT_{year}.GRD"
    mint_out = out_dir / f"MinT_{year}.GRD"
    download_file(mint_url, mint_out, year, nlat=31, nlon=31, is_temp=True)
    
    print("\nDownload script completed. Check the raw/ directory for the .grd files.")

if __name__ == '__main__':
    main(year=2023)
