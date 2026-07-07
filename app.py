from fastapi import FastAPI

from schemas import Passenger
from predictor import predict

app=FastAPI(
    title="Titanic survival API",
    version="1.0"
)

@app.get("/")
def home():
    return{
        "message": "Titanic prediction API is running"
    }

@app.post("/predict")
def predict_survival(passenger: Passenger):
    result = predict(passenger.model_dump())
    return result
