import torch
import torch.nn as nn
import math


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        timesteps,
        d_model=128,
        num_heads=4,
        num_layers=3,
        dropout=0.2
    ):
        super(TransformerModel, self).__init__()

        # --- Input Stem ---
        # LayerNorm estabiliza las escalas de las features (OHLCV, OBV, VIX…)
        self.input_norm = nn.LayerNorm(input_dim)

        # Conv1D extrae patrones locales (e.g., velas, momentum de 2-3 días)
        # antes de que el Transformer atienda a dependencias largas
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # --- Positional Encoding ---
        # Codificación posicional sinusoidal fija (más generalizable que nn.Parameter)
        self.register_buffer(
            'pos_encoding',
            self._build_sinusoidal_pe(timesteps, d_model)
        )
        self.pos_drop = nn.Dropout(dropout)

        # --- Transformer Encoder ---
        # dim_feedforward = 4 * d_model sigue la regla del paper original
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: gradientes más estables que Post-LN
            activation='gelu', # GELU supera a ReLU en la mayoría de Transformers modernos
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),  # Layer norm final sobre la secuencia
        )

        # --- Aggregation: CLS token ---
        # Un token [CLS] entrenable aprende a resumir toda la secuencia,
        # en lugar de promediar mecánicamente todos los timesteps
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- Clasificación Head ---
        # MLP de dos capas en lugar de una sola proyección lineal
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def _build_sinusoidal_pe(timesteps, d_model):
        """Codificación posicional sinusoidal (Vaswani et al., 2017)."""
        pe = torch.zeros(1, timesteps + 1, d_model)  # +1 para el CLS token
        position = torch.arange(0, timesteps + 1).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        B = x.size(0)

        # 1. Normalizar features de entrada
        x = self.input_norm(x)

        # 2. Conv1D stem: extrae patrones locales
        #    Conv1d espera (B, C, L) → transponer → (B, d_model, T) → transponer de vuelta
        x = self.conv_stem(x.transpose(1, 2)).transpose(1, 2)  # (B, T, d_model)

        # 3. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)            # (B, T+1, d_model)

        # 4. Positional encoding + dropout
        x = self.pos_drop(x + self.pos_encoding)

        # 5. Transformer Encoder
        x = self.transformer_encoder(x)

        # 6. Usar solo el CLS token para clasificación
        cls_out = x[:, 0]   # (B, d_model)

        # 7. MLP Head → logits
        return self.head(cls_out)

