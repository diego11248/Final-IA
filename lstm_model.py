
import torch
import torch.nn as nn

class FinBERTLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.2):
        super(FinBERTLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Capa LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, # Garantiza formato (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        # Capa densa de salida
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, timesteps, features)
        out, _ = self.lstm(x)
        
        # Tomamos solo el output del último timestep (proceso secuencial terminado)
        out = out[:, -1, :] 
        out = self.dropout(out)
        return self.fc(out)
