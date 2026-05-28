
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, timesteps, d_model=128, num_heads=4, num_layers=3, dim_feedforward=128, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        # Proyección lineal para ajustar las variables OHLCV a las dimensiones internas del Transformer
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Codificación Posicional Entrenable
        self.pos_embedding = nn.Parameter(torch.randn(1, timesteps, d_model))
        
        # Bloques de codificación del Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # Evita transponer (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Proyección y suma de posiciones
        x = self.embedding(x) + self.pos_embedding
        
        # Paso por el Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Global Average Pooling a lo largo de la dimensión temporal (timesteps)
        x = x.mean(dim=1)
        
        x = self.dropout(x)
        out = self.fc(x)
        return self.sigmoid(out)
