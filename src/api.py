from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import predict_customer


app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API for predicting term-deposit subscription",
    version="1.0.0"
)


class Customer(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


@app.get("/")
def home():
    return {
        "message": "Bank Marketing Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: Customer):
    try:
        result = predict_customer(
            customer.model_dump()
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )