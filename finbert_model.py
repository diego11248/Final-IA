import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class FineTunedFinBERT(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0.2):
        super(FineTunedFinBERT, self).__init__()
        
        # Cargar configuración y modelo FinBERT base (sin cabeza de clasificación NLP)
        # Esto descarga los pesos pre-entrenados para fine-tuning
        self.config = AutoConfig.from_pretrained("ProsusAI/finbert")
        self.finbert = AutoModel.from_pretrained("ProsusAI/finbert")
        
        # Congelar todas las capas base de FinBERT para evitar overfitting
        for param in self.finbert.parameters():
            param.requires_grad = False
            
        # Descongelar solo la última capa del encoder para fine-tuning fino
        for param in self.finbert.encoder.layer[-1].parameters():
            param.requires_grad = True
        
        # Proyectar características numéricas al tamaño oculto de FinBERT (768)
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Clasificador final 
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, timesteps, input_dim)
        
        # 1. Proyectar características numéricas a embeddings del tamaño de FinBERT
        # inputs_embeds shape: (batch_size, timesteps, 768)
        inputs_embeds = self.feature_projection(x)
        
        # 2. Pasar por las capas del Transformer FinBERT (Fine-Tuning)
        # Usamos inputs_embeds en lugar de input_ids ya que procesamos datos numéricos continuos.
        # FinBERT añadirá automáticamente los position_embeddings a esta secuencia temporal.
        outputs = self.finbert(inputs_embeds=inputs_embeds)
        
        # 3. Tomar el resumen secuencial (Mean Pooling en lugar de solo el último timestep)
        # last_hidden_state shape: (batch_size, timesteps, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        sequence_summary = torch.mean(last_hidden_state, dim=1) 
        
        # 4. Clasificación final binaria (Sube / Baja)
        sequence_summary = self.dropout(sequence_summary)
        logits = self.classifier(sequence_summary)
        
        return logits
