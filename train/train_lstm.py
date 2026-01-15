import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.stock_info_lstm import CandleLSTM
from dataloaders.aggregated_values import StockValuesDataLoader


def train_lstm(
    sequence_length=60,
    batch_size=128,
    hidden_size=128,
    num_layers=3,
    dropout=0.3,
    lr=0.001,
    max_epochs=20,
    normalization_method='percentage',
    num_workers=4,
    accelerator='auto',
    devices='auto',
    json_path='data_aggregated_v2/stock_values.json',
    stock_symbols=None,
    train_split=0.8
):
    """
    Train LSTM model on aggregated stock values
    
    Args:
        sequence_length: Number of past candles to use
        batch_size: Batch size for training
        hidden_size: LSTM hidden layer size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        lr: Learning rate
        max_epochs: Maximum number of epochs
        normalization_method: 'percentage', 'log', or 'minmax'
        num_workers: Number of data loading workers
        accelerator: 'auto', 'gpu', 'cpu', or 'mps'
        devices: Number of devices to use
        json_path: Path to stock_values.json file
        stock_symbols: List of stock symbols to use (None = all)
        train_split: Fraction of data to use for training
    """
    
    print("="*80)
    print("LSTM Training for Stock Price Prediction")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data Source: Aggregated JSON ({json_path})")
    if stock_symbols:
        print(f"  Stock Symbols: {', '.join(stock_symbols)}")
    else:
        print(f"  Stock Symbols: All available")
    print(f"  Sequence Length: {sequence_length} candles")
    print(f"  Batch Size: {batch_size}")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Num Layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning Rate: {lr}")
    print(f"  Max Epochs: {max_epochs}")
    print(f"  Normalization: {normalization_method}")
    print(f"  Train Split: {train_split:.1%}")
    print(f"  Accelerator: {accelerator}")
    print()
    
    # Initialize dataloader
    dataloader = StockValuesDataLoader(
        json_path=json_path,
        sequence_length=sequence_length,
        normalization_method=normalization_method,
        stock_symbols=stock_symbols,
        train_split=train_split
    )
    
    print(f"Loaded {len(dataloader.full_dataset)} sequences from aggregated data")
    print(f"  Training sequences: {len(dataloader.train_dataset)}")
    print(f"  Validation sequences: {len(dataloader.val_dataset)}")
    print()
    
    # Get dataloaders
    train_loader, val_loader = dataloader.get_train_val_loaders(
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Initialize model
    model = CandleLSTM(
        input_size=4,  # OHLCV
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr
    )
    
    print(f"Model Architecture:")
    print(model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_lstm_1d_4layers_256hidden',
        filename='lstm-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name='lstm_training'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train model
    print("="*80)
    print("Starting Training...")
    print("="*80)
    
    try:
        trainer.fit(model, train_loader, val_loader)
        
        print("\n" + "="*80)
        print("Training Completed!")
        print("="*80)
        print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {checkpoint_callback.best_model_score:.6f}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print(f"Last checkpoint saved at: {checkpoint_callback.last_model_path}")
    
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise
    
    return model, trainer, dataloader


def test_model(checkpoint_path, dataloader, num_samples=5, device='cpu'):
    """
    Test trained model with predictions
    
    Args:
        checkpoint_path: Path to model checkpoint
        dataloader: StockValuesDataLoader instance
        num_samples: Number of samples to test
        device: Device to run inference on
    """
    print("\n" + "="*80)
    print("Testing Model Predictions")
    print("="*80)
    
    # Load model
    model = CandleLSTM.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)
    
    # Get test data
    _, val_loader = dataloader.get_train_val_loaders(batch_size=1, num_workers=0)
    
    print(f"\nTesting {num_samples} samples...\n")
    
    for i, (x_batch, y_batch, contexts) in enumerate(val_loader):
        if i >= num_samples:
            break
        
        # Move data to device
        x_batch = x_batch.to(device)
        
        # Predict
        with torch.no_grad():
            y_pred = model(x_batch)
        
        # Denormalize (move back to CPU for numpy conversion)
        y_true_denorm = dataloader.denormalize(y_batch[0].cpu(), contexts[0])
        y_pred_denorm = dataloader.denormalize(y_pred[0].cpu(), contexts[0])
        
        print(f"Sample {i+1}:")
        print(f"  True:      O={y_true_denorm[0]:.2f} H={y_true_denorm[1]:.2f} "
              f"L={y_true_denorm[2]:.2f} C={y_true_denorm[3]:.2f}")
        print(f"  Predicted: O={y_pred_denorm[0]:.2f} H={y_pred_denorm[1]:.2f} "
              f"L={y_pred_denorm[2]:.2f} C={y_pred_denorm[3]:.2f}")
        
        # Calculate errors
        errors = abs(y_true_denorm - y_pred_denorm)
        pct_errors = (errors / (abs(y_true_denorm) + 1e-8)) * 100
        
        print(f"  Error %:   O={pct_errors[0]:.2f}% H={pct_errors[1]:.2f}% "
              f"L={pct_errors[2]:.2f}% C={pct_errors[3]:.2f}%")
        print()


if __name__ == "__main__":
    # Training configuration
    config = {
        'sequence_length': 12,        
        'batch_size': 32,
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.3,
        'lr': 0.0001,
        'max_epochs': 50,
        'normalization_method': 'percentage',
        'num_workers': 4,
        'accelerator': 'auto',
        'devices': 'auto',
        'json_path': 'data_aggregated_train/stock_values_all.json',
        'stock_symbols': None, #['AAPL', 'NVDA', 'AMZN', 'GOOGL', 'MSFT'],  # or None for all
        'train_split': 0.8
    }
    
    # Train model
    model, trainer, dataloader = train_lstm(**config)
    
    # Determine device for testing
    test_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test model if training completed successfully
    if trainer.checkpoint_callback.best_model_path:
        test_model(
            checkpoint_path=trainer.checkpoint_callback.best_model_path,
            dataloader=dataloader,
            num_samples=20,
            device=test_device
        )

    # test_model(
    #     checkpoint_path="/teamspace/studios/this_studio/TraderAgent/checkpoints_lstm_1d_16seq/lstm-epoch=25-val_loss=7.3272.ckpt",
    #     dataloader=dataloader,
    #     num_samples=20,
    #     device=test_device
    # )
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)