import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from dataset import ClimateDataset
from model import DigitalTwinPredictor
from torch.utils.data import DataLoader, Subset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    seq_length = 7
    target_var = 'rainfall'

    # 1. Data Ingestion & Fallback
    print("Loading data...")
    use_mock = True
    
    if use_mock:
        print("Using Mock Tensor Fallback for dry-run validation.")
        days = 366
        lats = np.linspace(6.5, 38.5, 129)
        lons = np.linspace(66.5, 100.0, 135)
        times = pd.date_range('2024-01-01', periods=days)
        
        # Generate mock spatial data tensor: (366, 129, 135)
        mock_data = np.random.rand(days, 129, 135).astype(np.float32)
        ds_merged = xr.Dataset({
            'rainfall': (['time', 'lat', 'lon'], mock_data),
            'min_temp': (['time', 'lat', 'lon'], mock_data),
            'max_temp': (['time', 'lat', 'lon'], mock_data)
        }, coords={'time': times, 'lat': lats, 'lon': lons})
    
    # 2. Dataset Initialization & Data Splitting
    full_dataset = ClimateDataset(ds_merged, sequence_length=seq_length, target_vars=[target_var])
    
    total_samples = len(full_dataset)
    # Contiguous Block Split with Sequence Buffer
    # First 80% for training
    train_end_idx = int(total_samples * 0.8) 
    buffer_len = seq_length
    # Drop buffer to prevent sequential leakage
    val_start_idx = train_end_idx + buffer_len
    
    train_indices = list(range(0, train_end_idx))
    # Ensure validation starts don't exceed total length
    val_indices = list(range(min(val_start_idx, total_samples), total_samples))
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # 3. Architecture Coupling
    model = DigitalTwinPredictor(
        input_channels=full_dataset.num_channels,
        hidden_channels=32,
        out_channels=len(full_dataset.target_vars)
    ).to(device)

    # 4. Robust Training Loop
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 5
    best_val_rmse = float('inf')
    
    # 5. State Checkpointing logic
    save_dir = Path('/content/drive/MyDrive/ISRO_Hackathon_PS5/models')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Suite
        model.eval()
        val_rmse_sum = 0.0
        val_mae_sum = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                predictions = model(X)
                
                # Spatial RMSE and MAE across (Height x Width)
                mse = torch.mean((y - predictions)**2).item()
                rmse = np.sqrt(mse)
                mae = torch.mean(torch.abs(y - predictions)).item()
                
                val_rmse_sum += rmse
                val_mae_sum += mae
                num_batches += 1
                
        avg_val_rmse = val_rmse_sum / max(num_batches, 1)
        avg_val_mae = val_mae_sum / max(num_batches, 1)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train MSE: {avg_train_loss:.4f} | Val RMSE: {avg_val_rmse:.4f} | Val MAE: {avg_val_mae:.4f}")
        
        # Checkpointing
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = save_dir / 'mugizhnokku_best.pth'
                torch.save(model.state_dict(), best_model_path)
                print(f"--> Saved best model to {best_model_path}")
            except Exception as e:
                # Local execution fallback
                local_path = Path('./models/mugizhnokku_best.pth')
                local_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), local_path)
                print(f"--> Local drive used. Saved best model to {local_path}")

    print("Training Orchestration Complete.")

if __name__ == '__main__':
    main()
