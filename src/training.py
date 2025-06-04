from sklearn.multioutput import MultiOutputClassifier
from sklearn_.linear_model import LogisticRegression
import joblib
import os

def train_model(x_train, y_train, save_path="models/model.pkl"):
    base_model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=10000)
    model = MultiOutputClassifier(base_model)
    model.fit(x_train, y_train)

    # Crea la carpeta si no existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Guarda el modelo entrenado
    joblib.dump(model, save_path)

    return model


def load_model(load_path="models/model.pkl"):
    return joblib.load(load_path)
