import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

def train(x,y, model_name='ml_models/linear_model.pkl'):
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    linear_regressor = pickle.load(open(model_name, 'rb'))
    linear_regressor.fit(x, y)
        
def predict(x, ml_model='ml_models/linear_model.pkl'):
    linear_regressor = pickle.load(open(ml_model, 'rb'))
    y_pred = linear_regressor.predict(np.array(x).reshape(-1,1))
    return y_pred
        
        