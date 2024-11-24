import pandas as pd
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)

class HousingPredictor:
    def __init__(self, model_path="models/best_model.pkl", scaler_path="models/scaler.pkl"):
        """Initialize the predictor with model and scaler"""
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Check if scaler exists
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

        # Load the model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load the scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Expected feature names in correct order
        self.expected_features = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]

    def validate_input(self, data: dict) -> None:
        """Validate input data"""
        # Check if all required features are present
        missing_features = set(self.expected_features) - set(data.keys())
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Check if data types are numeric
        for feature, value in data.items():
            if not isinstance(value[0], (int, float)):
                raise ValueError(f"Feature {feature} must be numeric, got {type(value[0])}")

    def preprocess_data(self, data: dict) -> pd.DataFrame:
        """Preprocess the input data"""
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Ensure correct column order
        df = df[self.expected_features]

        features_to_scale = [col for col in df.columns if col not in ['Latitude', 'Longitude']]

        # Create a copy of the original DataFrame
        df_scaled = df.copy()

        # Scale only the features that need scaling
        scaled_data = self.scaler.transform(df[features_to_scale])
        df_scaled[features_to_scale] = scaled_data

        return df_scaled

    def predict(self, data: dict) -> float:
        """Make prediction with preprocessed data"""
        try:
            # Validate input
            self.validate_input(data)

            # Preprocess data
            processed_data = self.preprocess_data(data)

            # Make prediction
            prediction = self.model.predict(processed_data)

            return prediction[0]

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
