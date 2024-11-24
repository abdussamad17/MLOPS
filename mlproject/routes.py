import logging


from fastapi import FastAPI
from pydantic import BaseModel

from modeltrain import CaliforniaHousingTrainer
from inference import HousingPredictor

logging.basicConfig(level=logging.INFO)

class House(BaseModel):
    MedInc: list[float]
    HouseAge: list[float]
    AveRooms: list[float]
    AveBedrms: list[float]
    Population: list[float]
    AveOccup: list[float]
    Latitude: list[float]
    Longitude: list[float]

app = FastAPI()
predictor = HousingPredictor()

base_url = "/api/v1"

@app.get(f"{base_url}/")
def index():
    return {"message": "House Price Prediction API"}

@app.get(f"{base_url}/health")
def health():
    return {"status": "OK"}

@app.post(f"{base_url}/predict")
def predict(data: House):
    logging.info(f"Received data: {data}")
    prediction = predictor.predict(data.model_dump())
    logging.info(f"Prediction: {prediction}")
    return {"House Value": float(prediction)}


@app.post(f"{base_url}/retrain")
async def retrain():
    trainer = CaliforniaHousingTrainer()
    trainer.train()
    logging.info("Model retraining has started. This process may take several minutes to complete.")
    return {"message": "Model retraining has started. This process may take several minutes to complete."}
