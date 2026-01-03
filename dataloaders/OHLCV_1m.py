import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from itertools import islice

class OHLCV1mStreamDataset(IterableDataset):
    """PyTorch IterableDataset for streaming OHLCV data with per-sequence normalization"""
    def __init__(self, hf_dataset, sequence_length=4, normalization_method='percentage'):
        """
        Args:
            hf_dataset: HuggingFace IterableDataset
            sequence_length: Number of past candles to use for prediction
            normalization_method: 'percentage' (percent change), 'log' (log returns), 
                                 'minmax' (per-sequence min-max), or None
        """
        self.hf_dataset = hf_dataset
        self.sequence_length = sequence_length
        self.normalization_method = normalization_method
        
    def normalize_sequence(self, sequence):
        """
        Normalize a sequence relative to itself
        
        Args:
            sequence: numpy array of shape (seq_len, 5) [open, high, low, close, volume]
            
        Returns:
            normalized_sequence, normalization_context (for denormalization)
        """
        if self.normalization_method == 'percentage':
            # Normalize as percentage change from first candle's close
            reference_price = sequence[0, 3]  # First candle's close
            reference_volume = sequence[:, 4].mean()  # Average volume
            
            normalized = sequence.copy()
            # Price columns (OHLC) as percentage change
            normalized[:, :4] = (sequence[:, :4] - reference_price) / reference_price * 100
            # Volume as percentage of average
            normalized[:, 4] = (sequence[:, 4] - reference_volume) / reference_volume * 100
            
            context = {
                'reference_price': reference_price,
                'reference_volume': reference_volume
            }
            
        elif self.normalization_method == 'log':
            # Log returns
            normalized = sequence.copy()
            # Prices: log(price_t / price_0)
            normalized[:, :4] = np.log(sequence[:, :4] / sequence[0, :4])
            # Volume: log ratio
            normalized[:, 4] = np.log(sequence[:, 4] / sequence[:, 4].mean() + 1e-8)
            
            context = {
                'first_prices': sequence[0, :4],
                'mean_volume': sequence[:, 4].mean()
            }
            
        elif self.normalization_method == 'minmax':
            # Min-max scaling per sequence
            min_vals = sequence.min(axis=0)
            max_vals = sequence.max(axis=0)
            range_vals = max_vals - min_vals
            range_vals = np.where(range_vals == 0, 1, range_vals)  # Avoid division by zero
            
            normalized = (sequence - min_vals) / range_vals
            
            context = {
                'min_vals': min_vals,
                'max_vals': max_vals
            }
            
        else:
            # No normalization
            normalized = sequence
            context = {}
        
        return normalized, context
        
    def __iter__(self):
        """Iterate through the dataset, yielding sequences"""
        buffer = []
        
        for example in self.hf_dataset:
            # Extract OHLCV values
            try:
                ohlcv = np.array([
                    float(example['open']),
                    float(example['high']),
                    float(example['low']),
                    float(example['close']),
                    float(example['volume'])
                ])
            except (KeyError, ValueError) as e:
                continue
            
            buffer.append(ohlcv)
            
            # Once we have enough candles, yield normalized sequence and target
            if len(buffer) > self.sequence_length:
                # Get sequence + target
                full_sequence = np.array(buffer[-self.sequence_length-1:])
                
                # Normalize the input sequence
                x_raw = full_sequence[:-1]
                y_raw = full_sequence[-1:]
                
                x_normalized, context = self.normalize_sequence(x_raw)
                
                # Normalize target using same context
                if self.normalization_method == 'percentage':
                    y_normalized = (y_raw - context['reference_price']) / context['reference_price'] * 100
                    y_normalized[:, 4] = (y_raw[:, 4] - context['reference_volume']) / context['reference_volume'] * 100
                elif self.normalization_method == 'log':
                    y_normalized = np.zeros_like(y_raw)
                    y_normalized[:, :4] = np.log(y_raw[:, :4] / context['first_prices'])
                    y_normalized[:, 4] = np.log(y_raw[:, 4] / context['mean_volume'] + 1e-8)
                elif self.normalization_method == 'minmax':
                    range_vals = context['max_vals'] - context['min_vals']
                    range_vals = np.where(range_vals == 0, 1, range_vals)
                    y_normalized = (y_raw - context['min_vals']) / range_vals
                else:
                    y_normalized = y_raw
                
                yield (
                    torch.FloatTensor(x_normalized), 
                    torch.FloatTensor(y_normalized.squeeze(0)),
                    context
                )


class OHLCV1mDataLoader:
    """DataLoader factory for 1-minute OHLCV data from HuggingFace (streaming)"""
    def __init__(self, sequence_length=4, dataset_name="mito0o852/OHLCV-1m", 
                 normalization_method='percentage'):
        """
        Args:
            sequence_length: Number of past candles to use for prediction
            dataset_name: HuggingFace dataset name
            normalization_method: 'percentage', 'log', 'minmax', or None
        """
        self.sequence_length = sequence_length
        self.dataset_name = dataset_name
        self.normalization_method = normalization_method
        
        print(f"Loading streaming dataset from HuggingFace: {dataset_name}")
        print(f"Normalization method: {normalization_method}")
        
        self.ds = load_dataset(self.dataset_name, streaming=True, split="train")
        print(f"Features: {self.ds.features}")
    
    def get_train_val_loaders(self, batch_size=32, num_workers=0):
        """
        Get PyTorch DataLoaders for training and validation (streaming)
        
        Args:
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for data loading
            
        Returns:
            train_loader, val_loader: PyTorch DataLoader objects
        """
        # Training dataset
        train_ds = load_dataset(self.dataset_name, streaming=True, split="train")
        train_dataset = OHLCV1mStreamDataset(
            train_ds, 
            self.sequence_length, 
            self.normalization_method
        )
        
        # Validation dataset
        val_ds = load_dataset(self.dataset_name, streaming=True, split="train")
        val_dataset = OHLCV1mStreamDataset(
            val_ds, 
            self.sequence_length, 
            self.normalization_method
        )
        
        print(f"\nCreating streaming dataloaders with batch_size={batch_size}")
        
        # Custom collate function to handle contexts
        def collate_fn(batch):
            x_batch = torch.stack([item[0] for item in batch])
            y_batch = torch.stack([item[1] for item in batch])
            contexts = [item[2] for item in batch]
            return x_batch, y_batch, contexts
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader
    
    def denormalize(self, normalized_data, context):
        """
        Denormalize data back to original scale using context
        
        Args:
            normalized_data: Normalized data (numpy array or torch tensor)
            context: Normalization context dict
            
        Returns:
            Denormalized data
        """
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
        
        if self.normalization_method == 'percentage':
            denorm = normalized_data.copy()
            denorm[:4] = (normalized_data[:4] / 100 * context['reference_price']) + context['reference_price']
            denorm[4] = (normalized_data[4] / 100 * context['reference_volume']) + context['reference_volume']
            return denorm
            
        elif self.normalization_method == 'log':
            denorm = normalized_data.copy()
            denorm[:4] = context['first_prices'] * np.exp(normalized_data[:4])
            denorm[4] = context['mean_volume'] * (np.exp(normalized_data[4]) - 1e-8)
            return denorm
            
        elif self.normalization_method == 'minmax':
            range_vals = context['max_vals'] - context['min_vals']
            return normalized_data * range_vals + context['min_vals']
            
        return normalized_data


if __name__ == "__main__":
    for method in ['percentage', 'log', 'minmax']:
        print(f"\n{'='*60}")
        print(f"Testing normalization method: {method}")
        print('='*60)
        
        dataloader = OHLCV1mDataLoader(
            sequence_length=12,
            dataset_name="mito0o852/OHLCV-1m",
            normalization_method=method
        )
        
        train_loader, val_loader = dataloader.get_train_val_loaders(batch_size=32)
        
        # Get one batch
        for x_batch, y_batch, contexts in train_loader:
            print(f"\nBatch shapes:")
            print(f"  Input: {x_batch.shape}")
            print(f"  Target: {y_batch.shape}")
            
            print(f"\nFirst sequence (normalized):")
            print(x_batch[0])
            print(f"\nFirst target (normalized):")
            print(y_batch[0])

            print(f"\nSecond sequence (normalized):")
            print(x_batch[1])
            print(f"\nSecond target (normalized):")
            print(y_batch[1])
            
            # Denormalize
            denorm = dataloader.denormalize(y_batch[0], contexts[0])
            print(f"\nDenormalized target:")
            print(denorm)
            
            break
        
        break  # Only test first method for quick check