import torch
from data_processing import prepare_data
from lstm_model import LSTMModel
from transformer_model import TransformerModel
from evaluation import train_and_evaluate

def main():
    # Configuración de hiperparámetros
    # Puedes expandir esta lista con hasta 500 tickers (e.g. componentes del S&P 500)
    TICKERS = [
        "AAPL", "MSFT", "NVDA", "AMD",  
        "JPM", "BAC", "WFC",            
        "AMZN", "TSLA", "HD",           
        "JNJ", "PFE", "LLY",            
        "XOM", "CVX"                    
    ]
    
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-31"
    TIMESTEPS = 30
    BATCH_SIZE = 128  # Aumentado para manejar mayor volumen de datos de manera eficiente
    EPOCHS =  100  # Ajustado para pruebas iniciales en múltiples tickers
    
    # 1. Carga y preprocesamiento de datos
    print(" Descargando y preparando DataLoaders de PyTorch...")
    train_loader, test_loader, timesteps, features, y_test = prepare_data(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        n_steps=TIMESTEPS,
        batch_size=BATCH_SIZE
    )
    
    # 2. Inicialización y entrenamiento de la LSTM
    lstm_net = LSTMModel(input_dim=features)
    train_and_evaluate(lstm_net, "Modelo LSTM (PyTorch)", train_loader, test_loader, y_test, epochs=EPOCHS, lr=0.01)
    
    # 3. Inicialización y entrenamiento del Transformer
    transformer_net = TransformerModel(input_dim=features, timesteps=timesteps)
    train_and_evaluate(transformer_net, "Modelo Transformer (PyTorch)", train_loader, test_loader, y_test, epochs=EPOCHS, lr=0.01)

if __name__ == "__main__":
    main()
