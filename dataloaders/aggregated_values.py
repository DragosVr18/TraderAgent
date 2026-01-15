
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class StockValuesDataset(Dataset):
    """Dataset for stock OHLCV values from aggregated JSON"""
    
    def __init__(self, json_path, sequence_length=12, normalization_method='percentage', stock_symbols=None):
        """
        Args:
            json_path: Path to stock_values.json file
            sequence_length: Number of past candles to use for prediction
            normalization_method: 'percentage', 'log', or 'minmax'
            stock_symbols: List of stock symbols to use (None = all available)
        """
        self.json_path = Path(json_path)
        self.sequence_length = sequence_length
        self.normalization_method = normalization_method
        
        # Load data
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter stock symbols if specified
        if stock_symbols is not None:
            self.data = {symbol: data for symbol, data in self.data.items() if symbol in stock_symbols}
        
        # Extract OHLCV sequences
        self.sequences = []
        self.create_sequences()
        
    def create_sequences(self):
        """Create sequences from raw stock data"""
        for symbol, candles in self.data.items():
            # Extract OHLCV values
            ohlcv_data = []
            for candle in candles:
                # Expected keys in each candle dict
                ohlcv = [
                    candle.get('Open', 0),
                    candle.get('High', 0),
                    candle.get('Low', 0),
                    candle.get('Close', 0),
                    # candle.get('Volume', 0)
                ]
                ohlcv_data.append(ohlcv)
            
            ohlcv_array = np.array(ohlcv_data, dtype=np.float32)
            
            # Create sequences (input: sequence_length candles, target: next candle)
            for i in range(len(ohlcv_array) - self.sequence_length):
                seq_x = ohlcv_array[i:i+self.sequence_length]
                seq_y = ohlcv_array[i+self.sequence_length]
                
                # Store with normalization context
                self.sequences.append({
                    'x': seq_x,
                    'y': seq_y,
                    'symbol': symbol,
                    'index': i
                })
    
    def normalize(self, data, context):
        """
        Normalize data using the specified method
        
        Args:
            data: numpy array of shape (seq_len, 5) or (5,)
            context: dict with normalization context
        """
        if self.normalization_method == 'percentage':
            # Normalize relative to first candle in sequence
            base_values = context['base_values']
            
            # Separate handling for Volume (index 4)
            normalized = data.copy()
            
            # OHLC normalization (indices 0-3)
            normalized[..., :4] = (data[..., :4] - base_values[:4]) / (base_values[:4] + 1e-8) * 100
            
            # Volume normalization - use log scale to prevent explosion
            # normalized[..., 4] = np.log1p(data[..., 4]) - np.log1p(base_values[4])
            
            return normalized
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
    
    def denormalize(self, data, context):
        """
        Denormalize data back to original scale
        
        Args:
            data: torch tensor or numpy array
            context: dict with normalization context
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        if self.normalization_method == 'percentage':
            base_values = context['base_values']
            denormalized = data.copy()
            
            # OHLC denormalization (indices 0-3)
            denormalized[..., :4] = (data[..., :4] / 100) * base_values[:4] + base_values[:4]
            
            # Volume denormalization - reverse log scale
            # denormalized[..., 4] = np.expm1(data[..., 4] + np.log1p(base_values[4]))
            
            return denormalized
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            x: tensor of shape (sequence_length, 5) - input sequence
            y: tensor of shape (5,) - target next candle
            context: dict with normalization context for denormalization
        """
        seq_data = self.sequences[idx]
        x = seq_data['x'].copy()
        y = seq_data['y'].copy()
        
        # Create normalization context
        if self.normalization_method == 'percentage':
            # Use first candle as base
            context = {'base_values': x[0].copy()}
        elif self.normalization_method == 'minmax':
            # Use min/max from sequence
            context = {
                'min_vals': x.min(axis=0),
                'max_vals': x.max(axis=0)
            }
        else:
            context = {}
        
        # Normalize
        x_norm = self.normalize(x, context)
        y_norm = self.normalize(y, context)
        
        # Convert to tensors
        x_tensor = torch.from_numpy(x_norm).float()
        y_tensor = torch.from_numpy(y_norm).float()
        
        return x_tensor, y_tensor, context


class StockValuesDataLoader:
    """Wrapper class for creating train/val dataloaders"""
    
    def __init__(self, json_path, sequence_length=60, normalization_method='percentage', 
                 stock_symbols=None, train_split=0.8):
        """
        Args:
            json_path: Path to stock_values.json file
            sequence_length: Number of past candles to use
            normalization_method: 'percentage', 'log', or 'minmax'
            stock_symbols: List of stock symbols to use (None = all)
            train_split: Fraction of data to use for training
        """
        self.json_path = json_path
        self.sequence_length = sequence_length
        self.normalization_method = normalization_method
        self.stock_symbols = stock_symbols
        self.train_split = train_split
        
        # Create full dataset
        self.full_dataset = StockValuesDataset(
            json_path=json_path,
            sequence_length=sequence_length,
            normalization_method=normalization_method,
            stock_symbols=stock_symbols
        )
        
        # Split into train/val
        self.create_train_val_split()
    
    def create_train_val_split(self):
        """Split dataset into training and validation sets"""
        total_size = len(self.full_dataset)
        train_size = int(total_size * self.train_split)
        val_size = total_size - train_size
        
        # Use random split
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to keep contexts as list of dicts
        
        Args:
            batch: List of tuples (x, y, context)
        
        Returns:
            x_batch: tensor of shape (batch_size, sequence_length, 5)
            y_batch: tensor of shape (batch_size, 5)
            contexts: list of context dicts
        """
        x_list, y_list, contexts = zip(*batch)
        
        x_batch = torch.stack(x_list)
        y_batch = torch.stack(y_list)
        
        # Keep contexts as list of dicts (don't let default collate merge them)
        return x_batch, y_batch, list(contexts)
    
    def get_train_val_loaders(self, batch_size=32, num_workers=4):
        """
        Create train and validation dataloaders
        
        Args:
            batch_size: Batch size for both loaders
            num_workers: Number of worker processes for data loading
        
        Returns:
            train_loader, val_loader
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=self.collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        
        return train_loader, val_loader
    
    def denormalize(self, data, context):
        """Helper method to denormalize predictions"""
        return self.full_dataset.denormalize(data, context)


# Example usage
if __name__ == "__main__":
    # Initialize dataloader
    dataloader = StockValuesDataLoader(
        json_path='data_aggregated_v2/stock_values.json',
        sequence_length=12,
        normalization_method='percentage',
        stock_symbols=['AAPL', 'NVDA', 'AMZN', 'GOOGL', 'MSFT']
    )
    
    print(f"Total sequences: {len(dataloader.full_dataset)}")
    print(f"Training sequences: {len(dataloader.train_dataset)}")
    print(f"Validation sequences: {len(dataloader.val_dataset)}")
    
    # Get dataloaders
    train_loader, val_loader = dataloader.get_train_val_loaders(batch_size=32, num_workers=4)
    
    # Test batch
    for x_batch, y_batch, contexts in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Input: {x_batch.shape}")  # (batch_size, sequence_length, 5)
        print(f"  Target: {y_batch.shape}")  # (batch_size, 5)
        print(f"  Contexts: {len(contexts)} contexts")

        # Print normalized first sample
        y_norm = y_batch[0].numpy()
        print(f"\nNormalized target (first sample):")
        print(f"  OHLCV: {y_norm}")
        
        # Test denormalization
        y_denorm = dataloader.denormalize(y_batch[0], contexts[0])
        print(f"\nDenormalized target (first sample):")
        print(f"  OHLCV: {y_denorm}")
        break