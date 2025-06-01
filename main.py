import os
import pandas as pd
import src.preprocessing as preprocessing
import src.training as training
import src.eval as evaluation

def main():
    df = pd.read_csv("data/source/tareas_hogar_dataset.csv")
    x_train, x_test, y_train, y_test = preprocessing.preprocess_data(df)

    model_path = "models/model.pkl"
    if os.path.exists(model_path):
        print("Cargando modelo desde disco...")
        model = training.load_model(model_path)
    else:
        print("Entrenando modelo desde cero...")
        model = training.train_model(x_train, y_train, save_path=model_path)

    evaluation.evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()
