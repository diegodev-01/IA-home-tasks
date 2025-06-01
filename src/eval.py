from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    # Si y_test es un DataFrame con dos columnas (time, difficult)
    accuracy_time = accuracy_score(y_test.iloc[:, 0], y_pred[:, 0])
    accuracy_difficult = accuracy_score(y_test.iloc[:, 1], y_pred[:, 1])

    print(f"Accuracy Time: {accuracy_time:.2f}")
    print(f"Accuracy Difficult: {accuracy_difficult:.2f}")

    print("\nClassification Report for Time:")
    print(classification_report(y_test.iloc[:, 0], y_pred[:, 0]))

    print("\nClassification Report for Difficult:")
    print(classification_report(y_test.iloc[:, 1], y_pred[:, 1]))

    return (accuracy_time, accuracy_difficult)
