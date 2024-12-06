import uvicorn
from fastapi import FastAPI, HTTPException
from models.passenger import Passenger
from libs.model import predict, train
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve(strict=True).parent
DATA_DIR = BASE_DIR / 'data'
MODEL_ML_DIR = BASE_DIR / 'ml_models'

app = FastAPI()

@app.get("/", tags=["intro"])
async def index():
    return {"message": "Titanic Survival Prediction API"}

@app.post("/model/train", tags=["model"], status_code=200)
async def train_model(data_name="DSP_6", model_name="logistic_model"):
    data_file = DATA_DIR / f"{data_name}.csv"
    model_file = MODEL_ML_DIR / f"{model_name}.pkl"

    if not data_file.exists():
        raise HTTPException(status_code=404, detail="Training data not found")

    df = pd.read_csv(data_file)

    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df.drop(columns=['Cabin'], inplace=True)
    df.dropna(inplace=True)

    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    train(X, y, model_file)
    return {"model_fit": "success", "model_save": "success"}

@app.post("/model/predict", tags=["model"], status_code=200)
async def get_predictions(passenger: Passenger, model_name="logistic_model"):
    model_file = MODEL_ML_DIR / f"{model_name}.pkl"
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    data = passenger.dict()
    df = pd.DataFrame([data])

    df['Age'].fillna(df['Age'].mean(), inplace=True)
    if 'Cabin' in df.columns:
        df.drop(columns=['Cabin'], inplace=True)
    df.dropna(inplace=True)

    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    expected_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]

    y_pred = predict(df, ml_model=model_file)
    return {"prediction": int(y_pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)