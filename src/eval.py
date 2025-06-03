from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def format_report(report, mapping):
    """
    Convierte el reporte (output_dict=True) en un DataFrame y renombra los índices
    (las clases) usando el diccionario de mapeo, de forma que en lugar de ver números se muestren las etiquetas originales.
    Se eliminan filas globales como 'accuracy'.
    """
    df = pd.DataFrame(report).transpose().drop(['accuracy'], errors='ignore')
    
    # Creamos un nuevo índice usando el mapping; si la fila es un número, lo reemplazamos.
    new_index = {}
    for idx in df.index:
        try:
            # Convertimos el índice a entero y buscamos su etiqueta en mapping
            new_index[idx] = mapping[int(idx)]
        except (ValueError, KeyError):
            # Si no se puede, se deja el índice tal cual (para 'macro avg', 'weighted avg', etc.)
            new_index[idx] = idx
    df.rename(index=new_index, inplace=True)
    return df

def evaluate_model(model, x_test, y_test, mapping_time, mapping_difficult):
    # Obtener las predicciones
    y_pred = model.predict(x_test)
    
    # Calcular la accuracy para cada objetivo
    accuracy_time = accuracy_score(y_test.iloc[:, 0], y_pred[:, 0])
    accuracy_difficult = accuracy_score(y_test.iloc[:, 1], y_pred[:, 1])
    
    print(f"Accuracy Time: {accuracy_time:.2f}")
    print(f"Accuracy Difficult: {accuracy_difficult:.2f}")
    
    # Generar los classification reports como diccionarios
    report_time = classification_report(y_test.iloc[:, 0], y_pred[:, 0], output_dict=True)
    report_difficult = classification_report(y_test.iloc[:, 1], y_pred[:, 1], output_dict=True)
    
    print("\nClassification Report for Time:")
    print(classification_report(y_test.iloc[:, 0], y_pred[:, 0]))
    print("\nClassification Report for Difficult:")
    print(classification_report(y_test.iloc[:, 1], y_pred[:, 1]))
    
    # Convertir los reportes a DataFrame y formatearlos utilizando los mappings
    df_time = format_report(report_time, mapping_time)
    df_difficult = format_report(report_difficult, mapping_difficult)
    
    # 1. Gráfica de barras para las métricas por clase - Time
    plt.figure(figsize=(10,6))
    df_time[['precision', 'recall', 'f1-score']].plot(kind='bar', rot=0)
    plt.title('Métricas por Clase - Time')
    plt.xlabel('Clase')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # 2. Gráfica de barras para las métricas por clase - Difficult
    df_difficult[['precision', 'recall', 'f1-score']].plot(kind='bar', rot=0,
                                                           color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Métricas por Clase - Difficult')
    plt.xlabel('Clase')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # 3. Gráfica comparativa de Accuracy Global por Objetivo
    plt.bar(['Time', 'Difficult'], [accuracy_time, accuracy_difficult], color=['#9467bd', '#8c564b'])
    plt.title('Accuracy Global por Objetivo')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    return (accuracy_time, accuracy_difficult)
