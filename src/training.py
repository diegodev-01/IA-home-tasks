from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def train_model(x_train, y_train):
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = MultiOutputClassifier(base_model)
    model.fit(x_train, y_train)
    return model
