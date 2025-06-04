
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn_.linear_model import LogisticRegression
import src.preprocessing as preprocessing

# 1. Cargar y preprocesar los datos reales del dataset
csv_path = "data/source/tareas_hogar_dataset.csv"
df = pd.read_csv(csv_path)
x_train, x_test, y_train, y_test, mapping_time, mapping_difficult = preprocessing.preprocess_data(df)

# 2. Entrenar el modelo igual que en src/training.py
base_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100 )
model = MultiOutputClassifier(base_model)
model.fit(x_train, y_train)

# 3. Mostrar los "logits" (puntuaciones) para cada salida
print("\n=== PUNTUACIONES (LOGITS) POR CLASE ===")
for output_idx, (est, mapping, nombre) in enumerate([
    (model.estimators_[0], mapping_time, 'time'),
    (model.estimators_[1], mapping_difficult, 'difficult')
]):
    sample = x_test.iloc[5] 
    logits = np.array([regre.predict([sample])[0] for regre in est.estimators_])
    from collections import Counter
    logit_counts = Counter(logits)
    print(f"\nEntrada de test (muestra ejemplo) para output '{nombre}': {sample.to_dict()}")
    print(f"Logits por clase ({nombre}):")
    for clase_cod, count in logit_counts.items():
        clase_real = mapping.get(clase_cod, clase_cod)
        print(f"  Clase codificada: {clase_cod}  |  Clase real: {clase_real}  |  Logits: {count}")

# 4. Mostrar la suma de probabilidades
print("\n=== PROBABILIDADES SOFTMAX PARA LA MUESTRA EJEMPLO ===")
for i, est in enumerate(model.estimators_):
    # Convierte la muestra a DataFrame para mantener los nombres de columnas
    muestra = pd.DataFrame([sample], columns=x_test.columns)
    probs = est.predict_proba(muestra)[0]
    mapping = mapping_time if i == 0 else mapping_difficult
    clases = [mapping.get(idx, idx) for idx in range(len(probs))]
    nombre = 'time' if i == 0 else 'difficult'
    print(f"\nOutput {i} ({nombre}):")
    for clase, prob in zip(clases, probs):
        print(f"  Clase: {clase}  |  Probabilidad softmax: {prob:.4f}")
    print(f"  Suma total de probabilidades softmax: {np.sum(probs):.4f}")

print("\nDemostración completa con datos reales: logits, probabilidades softmax.\n")
