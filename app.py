import uvicorn
from fastapi import FastAPI, HTTPException, Form

from models.point import Point
from libs.model import predict, train

from pathlib import Path

from typing import Annotated


BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = Path(BASE_DIR).joinpath('models')
DATA_DIR = Path(BASE_DIR).joinpath('data')
MODEL_ML_DIR = Path(BASE_DIR).joinpath('ml_models')

app = FastAPI()

@app.get("/", tags=["intro"])
async def index():
    return {"message": "Linear regression model API"}

@app.post("/model/point", tags=["data"], response_model=Point, status_code=200)
async def point(x: Annotated[int, Form()], y: Annotated[float, Form()]):
    return Point(x=x, y=y)

@app.post("/model/train", tags=["model"], status_code=200)
async def train_model(data: Point, data_name="10_points", model_name="linear_model"):
    data_file = Path(DATA_DIR).joinpath(f"{data_name}.csv")
    model_file = Path(MODEL_ML_DIR).joinpath(f"{model_name}.pkl")
    
    data = data.model_dump()
    x = data['x']
    y = data['y']
     
    train(x,y,model_file)
    
    response_object = {"model_fit": "success", "model_save": "success"}
    return response_object

@app.post("/model/predict", tags=["model"], status_code=200)
async def get_predictions(data:Point, model_name="linear_model"):
    model_file = Path(MODEL_ML_DIR).joinpath(f"{model_name}.pkl")
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    data = data.model_dump()
    x = data['x']
    
    y_pred = predict(x=x, ml_model=model_file)
    data['y'] = y_pred
    
    response_object = {"x": x, "y": y_pred[0][0]}
    return response_object

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)