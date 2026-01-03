import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

class CandleLSTM(pl.LightningModule):
    """LSTM model for predicting next candle from previous candles"""
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Dropout only if num_layers > 1
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, input_size)  # Output: [open, high, low, close, volume]
        self.lr = lr
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent exploding/vanishing gradients"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction
    
    def training_step(self, batch, batch_idx):
        x, y, contexts = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        
        # Check for NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss detected at batch {batch_idx}")
            print(f"  y_pred stats: min={y_pred.min():.2f}, max={y_pred.max():.2f}, mean={y_pred.mean():.2f}")
            print(f"  y stats: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
            return None
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, contexts = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)
        
        # Check for NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid validation loss detected at batch {batch_idx}")
            return None
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log additional metrics
        mae = nn.functional.l1_loss(y_pred, y)
        self.log('val_mae', mae, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Use learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def predict_next_candle(self, last_4_candles):
        """
        Predict next candle from last 4 candles
        Args:
            last_4_candles: numpy array of shape (4, 5) with OHLCV data
        Returns:
            numpy array of shape (5,) with predicted OHLCV values
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(last_4_candles).unsqueeze(0)  # Add batch dimension
            prediction = self(x)
        return prediction.squeeze(0).numpy()
