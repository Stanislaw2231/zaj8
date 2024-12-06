import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train(X, y, model_name='ml_models/logistic_model.pkl'):
    logistic_regressor = LogisticRegression(max_iter=1000)
    logistic_regressor.fit(X, y)
    with open(model_name, 'wb') as file:
        pickle.dump(logistic_regressor, file)

def predict(X, ml_model='ml_models/logistic_model.pkl'):
    with open(ml_model, 'rb') as file:
        logistic_regressor = pickle.load(file)
    y_pred = logistic_regressor.predict(X)
    return y_pred