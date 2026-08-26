import joblib
import pandas as pd
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the saved model
MODEL_PATH = PROJECT_ROOT / "models" / "bank_marketing_model.joblib"


# Load the trained model artifact
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]


def predict_customer(data: dict):
    """
    Predict whether a customer is likely to subscribe
    to the bank's term deposit.
    """

    input_data = pd.DataFrame([data])

    # Ensure the model receives exactly the expected features
    input_data = input_data[features]

    # Get probability of positive class (yes)
    probability = model.predict_proba(input_data)[0][1]

    # Apply the selected threshold
    prediction = "yes" if probability >= threshold else "no"

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "threshold": threshold
    }