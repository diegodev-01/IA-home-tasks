import pandas as pd
import src.preprocessing as preprocessing
import src.training as training
import src.eval as evaluation  # este es el módulo

def main():
    df = pd.read_csv("data/source/tareas_hogar_dataset.csv")
    x_train, x_test, y_train, y_test = preprocessing.preprocess_data(df)
    model = training.train_model(x_train, y_train)
    evaluation.evaluate_model(model, x_test, y_test)  # Aquí llamas a la función dentro del módulo

if __name__ == "__main__":
    main()
