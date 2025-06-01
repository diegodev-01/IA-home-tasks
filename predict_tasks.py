import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Carga el modelo entrenado
model = joblib.load("models/model.pkl")

# Recrea los LabelEncoders con las clases reales usadas en entrenamiento

le_task = LabelEncoder()
le_task.classes_ = np.array([
    "Barrer", "Cocinar", "Lavar platos", "Lavar ropa", "Limpiar baño",
    "Limpiar ventanas", "Organizar sala", "Planchar ropa", "Regar plantas",
    "Sacar la basura", "Tender cama"
])

le_priority = LabelEncoder()
le_priority.classes_ = np.array(["Baja", "Alta", "Media"])

le_weather = LabelEncoder()
le_weather.classes_ = np.array(["Lluvioso", "Nublado", "Soleado", "Ventoso"])

le_hour = LabelEncoder()
le_hour.classes_ = np.array(["Mañana", "Noche", "Tarde"])

le_time = LabelEncoder()
le_time.classes_ = np.array([10, 15, 20, 30, 45, 60, 75, 90])

le_difficult = LabelEncoder()
le_difficult.classes_ = np.array([1, 2, 3, 4, 5])

# Recibe tareas por línea de comando separadas por coma
input_tasks = sys.argv[1] if len(sys.argv) > 1 else ""
tasks_list = [t.strip() for t in input_tasks.split(",") if t.strip()]

if not tasks_list:
    print("No se proporcionaron tareas para predecir.")
    sys.exit(1)

# Valores fijos para las otras columnas (puedes cambiar si quieres)
default_priority = "Media"
default_weather = "Soleado"
default_hour = "Mañana"

# Crear dataframe para predecir
df_pred = pd.DataFrame({
    "task": tasks_list,
    "priority": [default_priority] * len(tasks_list),
    "weather": [default_weather] * len(tasks_list),
    "hour": [default_hour] * len(tasks_list)
})

# Codificar características con LabelEncoders
try:
    df_pred["task"] = le_task.transform(df_pred["task"].values)
    df_pred["priority"] = le_priority.transform(df_pred["priority"].values)
    df_pred["weather"] = le_weather.transform(df_pred["weather"].values)
    df_pred["hour"] = le_hour.transform(df_pred["hour"].values)
except ValueError as e:
    print(f"Error al codificar: {e}")
    print("Revisa que las tareas y valores sean válidos y estén en las clases definidas.")
    sys.exit(1)

# Predicción
predictions = model.predict(df_pred)

# Decodificar resultados
time_preds = le_time.inverse_transform(predictions[:, 0])
difficult_preds = le_difficult.inverse_transform(predictions[:, 1])

# Crear DataFrame con resultados
results = pd.DataFrame({
    "task": tasks_list,
    "predicted_time": time_preds,
    "predicted_difficult": difficult_preds
})

# Ordenar por dificultad (de más fácil a más difícil)
order = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
results["difficult_order"] = results["predicted_difficult"].map(order)
results = results.sort_values("difficult_order").drop(columns=["difficult_order"])

print(results.to_string(index=False))
