import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import sys
import os

# Parámetro de umbral
THRESHOLD = 5000.0

# --- Configurar MLflow igual que en train.py ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
mlflow.set_tracking_uri("file://" + os.path.abspath(mlruns_dir))

# --- Cargar dataset ---
print("--- Debug: Cargando dataset load_diabetes ---")
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---")

# --- Cargar modelo desde MLflow ---
print("--- Debug: Intentando cargar modelo desde MLflow ---")
print("--- Debug: Intentando cargar modelo desde MLflow ---")
print("--- Debug: Intentando cargar modelo desde MLflow ---")
print("--- Debug: Intentando cargar modelo desde MLflow ---")

try:
    experiment_name = "CI-CD-Lab2"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise Exception(f"Experimento '{experiment_name}' no encontrado")
    
    # Obtener el último run
    runs = mlflow.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
    
    if runs.empty:
        raise Exception("No se encontraron runs en el experimento")
    
    run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"--- Debug: Cargando modelo desde URI: {model_uri} ---")
    
    model = mlflow.sklearn.load_model(model_uri)
    print("--- Debug: Modelo cargado exitosamente desde MLflow ---")

except Exception as e:
    print(f"--- ERROR al cargar modelo desde MLflow: {str(e)} ---")
    print(f"--- Debug: Archivos en {os.getcwd()}: ---")
    print(os.listdir(os.getcwd()))
    sys.exit(1)

# --- Predicción y Validación ---
print("--- Debug: Realizando predicciones ---")
try:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"🔍 MSE del modelo: {mse:.4f} (umbral: {THRESHOLD})")

    # Validación
    if mse <= THRESHOLD:
        print(" El modelo cumple los criterios de calidad.")
        sys.exit(0)
    else:
        print(" El modelo no cumple el umbral. Deteniendo pipeline.")
        sys.exit(1)

except Exception as pred_err:
    print(f"--- ERROR durante la predicción: {pred_err} ---")
    if hasattr(model, 'n_features_in_'):
        print(f"Modelo esperaba {model.n_features_in_} features.")
    print(f"X_test tiene {X_test.shape[1]} features.")
    sys.exit(1)