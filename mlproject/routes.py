import logging

from fastapi import FastAPI, HTTPException
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
predictor = None  # Initialize as None

base_url = "/api/v1"

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = HousingPredictor()
        except FileNotFoundError:
            # Model hasn't been trained yet
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please train the model first using the /retrain endpoint."
            )
    return predictor

@app.get(f"{base_url}/")
def index():
    return {"message": "House Price Prediction API"}

@app.get(f"{base_url}/health")
def health():
    try:
        # Try to load the predictor
        get_predictor()
        return {"status": "OK", "model": "loaded"}
    except HTTPException:
        # Model files not found, but service is still running
        return {"status": "OK", "model": "not loaded"}

@app.post(f"{base_url}/predict")
def predict(data: House):
    logging.info(f"Received data: {data}")
    predictor = get_predictor()  # Get or initialize predictor
    prediction = predictor.predict(data.model_dump())
    logging.info(f"Prediction: {prediction}")
    return {"House Value": float(prediction)}

@app.post(f"{base_url}/retrain")
async def retrain():
    trainer = CaliforniaHousingTrainer()
    trainer.train()
    logging.info("Model retraining has started. This process may take several minutes to complete.")
    return {"message": "Model retraining has started. This process may take several minutes to complete."}
