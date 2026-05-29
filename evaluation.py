import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def train_and_evaluate(model, name_model, train_loader, val_loader, test_loader, y_true_test, epochs=30, lr=0.001):
    print(f"\n Entrenando: {name_model}")
    
    # Configuración de dispositivo (Usa GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss() # Función de pérdida binaria con logits
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler: Reduce el learning rate a la mitad cada 5 épocas
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    
    # Bucle de Entrenamiento
    best_loss = float('inf')
    patience = 5
    stale_epochs = 0
    min_delta = 1e-4
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Resetear gradientes
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass y optimización
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
            
        # Actualizar el learning rate
        scheduler.step()
        
        total_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(total_train_loss)
        
        # --- NUEVO: Calcular pérdida de validación ---
        model.eval()
        val_loss = 0.0
        if val_loader is not None:
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs_val = model(X_val)
                    v_loss = criterion(outputs_val, y_val)
                    val_loss += v_loss.item() * X_val.size(0)
            
            total_val_loss = val_loss / len(val_loader.dataset)
        else:
            total_val_loss = total_train_loss # Fallback si no hay val_loader
            
        val_losses.append(total_val_loss)
        
        # Early Stopping usando val_loss
        if total_val_loss < best_loss - min_delta:
            best_loss = total_val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            
        if (epoch + 1) % 5 == 0 or epoch == 0 or stale_epochs >= patience:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Época [{epoch+1}/{epochs}] - Train Loss: {total_train_loss:.4f} - Val Loss: {total_val_loss:.4f} - LR: {current_lr:.6f}")
            
        if stale_epochs >= patience:
            print(f"\n[Early Stopping] Entrenamiento detenido en la época {epoch+1}.")
            print(f"La pérdida de validación no ha mejorado en {patience} épocas (se volvió 'stale').")
            break
            
    # --- Generar y guardar gráfica de pérdidas ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='orange')
    plt.title(f'Curvas de Aprendizaje: {name_model}')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (BCEWithLogitsLoss)')
    plt.legend()
    plt.grid(True)
    
    # Limpiar el nombre para que sea válido para archivo
    safe_name = "".join([c if c.isalnum() else "_" for c in name_model])
    plt.savefig(f'learning_curve_{safe_name}.png')
    plt.close()
    print(f">>> Curva de aprendizaje guardada como 'learning_curve_{safe_name}.png' <<<")
            
    # Bucle de Evaluación
    model.eval()
    all_outputs = []
    
    with torch.no_grad(): # Desactivar cálculo de gradientes para ahorrar memoria
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs) # Aplicar sigmoide para obtener probabilidades
            all_outputs.append(probs.cpu().numpy())
            
    y_raw = np.vstack(all_outputs)
    
    # Calcular el umbral dinámico basado en la mediana de las predicciones del test set
    threshold = np.median(y_raw)
    y_pred = (y_raw > threshold).astype(int)
    
    # Despliegue de métricas
    print("\n" + "="*50)
    print(f"      REPORTE DE EVALUACIÓN: {name_model.upper()}      ")
    print("="*50)
    print(f"Umbral de decisión dinámico (mediana): {threshold:.4f}")
    print("-"*50)
    print(classification_report(y_true_test, y_pred, target_names=['Baja', 'Subida'], zero_division=0))
    
    print("MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_true_test, y_pred))
    print("="*50 + "\n")
