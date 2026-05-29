# Proyecto Final IA: Comparacion de LSTM vs. Transformers para la Prediccion de Acciones

Este proyecto analiza el funcionamiento y la efectividad de usar redes neuronales recurrentes (LSTM) frente a modelos basados en Transformers (FinBERT) para predecir el comportamiento de una accion en terminos binarios (subida o bajada).

## Funcionamiento del Sistema

El pipeline del sistema esta estructurado en cinco fases consecutivas que abarcan desde la adquisicion de datos hasta la inferencia en tiempo real:

### 1. Adquisicion de Datos y Extraccion de Sentimiento
*   **Datos Historicos:** Se descargan precios diarios (OHLCV) de multiples acciones utilizando la biblioteca `yfinance`.
*   **Analisis de Sentimiento:** Se procesan textos de noticias (sintetizadas o reales) mediante el modelo preentrenado **FinBERT** (`ProsusAI/finbert`) para extraer probabilidades continuas de sentimiento positivo, negativo y neutral.
*   **Indicadores Tecnicos:** Se calculan indicadores tecnicos adicionales como los retornos rezagados y las Bandas de Bollinger a partir de los precios de cierre.

### 2. Preparacion de Datos y Prevencion de Data Leakage
*   **Escalamiento Seguro:** Se entrena un escalador `MinMaxScaler` por cada accion utilizando unicamente los limites del conjunto de entrenamiento de cada fold. Los parametros del escalador final se exportan en `scalers.pkl`.
*   **Ventanas Temporales:** Los datos continuos se estructuran en secuencias de 30 dias de duracion (timesteps) con formato `(lote, 30, 10)`, donde 10 es el numero de caracteristicas finales (precios, volumen, indicadores tecnicos y vectores de sentimiento).

### 3. Modelos de Aprendizaje Profundo
*   **Modelo FinBERT + LSTM:** Procesa la secuencia de entrada mediante un LSTM de dos capas. Extrae la representacion del ultimo paso de tiempo (timestep final) y la clasifica mediante una capa densa para predecir la direccion del mercado.
*   **Modelo Fine-Tuned FinBERT:** Utiliza una capa de proyeccion lineal para mapear las caracteristicas de entrada a la dimension oculta de FinBERT (768). Esta secuencia se introduce directamente como embeddings al Transformer FinBERT (con su ultima capa de codificador descongelada para ajuste fino). Finalmente, se consolida la informacion temporal mediante un promedio de la secuencia completa (Mean Pooling) antes de clasificar.

### 4. Evaluacion y Entrenamiento de Produccion
*   **Validacion Walk-Forward:** Se evalua la robustez del sistema a lo largo del tiempo dividiendo el conjunto de datos en tres pliegues (folds) temporales sucesivos, evitando el sesgo de mirar al futuro.
*   **Umbral Dinamico:** La clasificacion no utiliza un limite rigido de 0.5, sino que establece el umbral basandose en la mediana de las predicciones del test set para adaptarse al sesgo de prediccion del modelo.
*   **Guardado:** Al finalizar el entrenamiento de produccion, los pesos se exportan a los archivos `finbert_lstm_model.pt` y `finetuned_finbert_model.pt`.

### 5. Inferencia y Consenso
*   El script `predict.py` descarga de forma interactiva los ultimos 120 dias de datos de un ticker, reconstruye las caracteristicas de la ventana mas reciente de 30 dias, las normaliza con el escalador guardado y calcula las probabilidades individuales de cada modelo.
*   La prediccion final se calcula promediando los resultados de ambos modelos (Consenso) para emitir una señal binaria (Bullish/Bearish) acompañada de su nivel de confianza estimado.

### 6. Uso del modelo
* el modelo requiere que se corra main.py antes de hacer predicciones. main.py entrena el modelo y guarda los pesos en finbert_lstm_model.pt y finetuned_finbert_model.pt.
* Una vez entrenado el modelo, se puede utilizar el script `predict.py` que descarga los ultimos 120 dias de datos de un ticker y calcula la prediccion final.

---

## Declaracion de uso de IA
El modelo base se implemento utilizando la ayuda del agente integrado de Antigravity para corregir formato y bugs.
