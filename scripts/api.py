from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

CLEAN_CSV_PATH = "./data/clean/clean_data.csv"
MODEL = "model/request_classifier_pipeline.joblib"

model = joblib.load(MODEL)

def get_categories():
    df = pd.read_csv(CLEAN_CSV_PATH)
    categories = df["category"].drop_duplicates().to_list()

    return categories

app = FastAPI()

class PredicitonRequest(BaseModel):
    request_text: str

@app.get("/health")
async def root():
    return {"status" : "ok"}

@app.get("/categories")
async def root():
    categories = get_categories()
    return {"categories: ": categories}

@app.post("/predict/")
async def make_prediction(request: PredicitonRequest):
    request_data = [request.request_text]

    prediction = model.predict(request_data)

    probabilities = model.predict_proba(request_data)

    highest_prob_index = np.argmax(probabilities[0])
    confidence_score = float(probabilities[0][highest_prob_index])

    return {"request_text": request.request_text,
            "predicted_category": prediction[0],
            "confidence_score": confidence_score
            }