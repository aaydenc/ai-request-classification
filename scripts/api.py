from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import joblib
import numpy as np

CLEAN_CSV_PATH = "./data/clean/clean_data.csv"
MODEL = "model/request_classifier_pipeline.joblib"

pipeline = joblib.load(MODEL)

def get_categories() -> list[str]:
    """ Reads the CSV file and returns all the unique categories

    Returns: 
        A list containing the unique categories.
    """
    df = pd.read_csv(CLEAN_CSV_PATH)
    categories = df["category"].dropna().drop_duplicates().to_list()

    return categories

app = FastAPI(title="Public Request Classification API",
    description="An API for classifying public service requests.",
    version="1.0.0")
categories = get_categories()

class PredictionRequest(BaseModel):
    request_text: str = Field(min_length=1)

    @field_validator("request_text")
    @classmethod
    def validate_request_text(cls, value):
        if not value.strip():
            raise ValueError("Request text cannot be empty")

        return value

class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    request_text: str
    predicted_category: str
    confidence_score: float


@app.get("/health")
def health_check():
    return {"status" : "ok"}


@app.get("/categories")
def list_categories():
    return {"categories": categories}


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    request_data = [request.request_text]

    prediction = pipeline.predict(request_data)

    probabilities = pipeline.predict_proba(request_data)

    confidence_score = float(probabilities[0].max())

    return {"request_text": request.request_text,
            "predicted_category": prediction[0],
            "confidence_score": confidence_score
            }