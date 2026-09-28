from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("model/linear_regression_model.joblib")


class InputData(BaseModel):
    MedInc: float
    Latitude: float
    Longitude: float
    AveRooms: float
    HouseAge: float

class PredictionResponse(BaseModel):
    predicted_house_price: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    input_data = np.array([[
    data.MedInc,
    data.Latitude,
    data.Longitude,
    data.AveRooms,
    data.HouseAge
    ]])

    prediction = model.predict(input_data)

    return {
        "predicted_house_price": float(prediction[0])
    }