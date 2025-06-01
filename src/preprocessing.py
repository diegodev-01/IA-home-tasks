import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Codificadores para las columnas categóricas
    le_priority = LabelEncoder()
    le_weather = LabelEncoder()
    le_hour = LabelEncoder()
    le_task = LabelEncoder()
    le_time = LabelEncoder()
    le_difficult = LabelEncoder()

    # Codifica las columnas categóricas que son features
    df["priority"] = le_priority.fit_transform(df["priority"])
    df["weather"] = le_weather.fit_transform(df["weather"])
    df["hour"] = le_hour.fit_transform(df["hour"])
    df["task"] = le_task.fit_transform(df["task"])

    # Codifica las columnas objetivo (targets)
    df["time"] = le_time.fit_transform(df["time"])
    df["difficult"] = le_difficult.fit_transform(df["difficult"])

    # Features (X): task, priority, weather, hour
    X = df[["task", "priority", "weather", "hour"]]

    # Targets (y): time y difficult
    y = df[["time", "difficult"]]

    # Divide en entrenamiento y prueba (multi-output)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test
