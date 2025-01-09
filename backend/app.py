from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import uvicorn

app = FastAPI()
# Import the existing functions from your backend
from model_construction import organize_data, train_data, save_model, predict, evaluate_model

# Define paths
DATA_DIR = "data/dataset.csv"
COMPLETE_MODEL_DIR = "models/complete_model.joblib"


# Pydantic model for input text
class PredictionRequest(BaseModel):
    text: List[str]  # Accepts a list of text inputs for predictions


@app.post("/train")
def train_model():
    """Endpoint to train the model."""
    if not os.path.exists(DATA_DIR):
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        # Train the model
        data = organize_data(DATA_DIR)
        model, vectorizer, X_test_vectorized, y_test = train_data(data)

        # Evaluate the model
        evaluate_model(model, X_test_vectorized, y_test)

        # Save the model and vectorizer
        save_model(model, vectorizer, COMPLETE_MODEL_DIR)

        return {"message": "Model trained and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict_text(request: PredictionRequest):
    """Endpoint to make predictions on input text."""
    if not os.path.exists(COMPLETE_MODEL_DIR):
        raise HTTPException(status_code=404, detail="Trained model not found.")

    label_mapping = {1: "FAKE", 0: "REAL"}

    try:
        predictions = predict(request.text)
        # Zamiana 0 i 1 na odpowiednie nazwy
        mapped_predictions = [label_mapping[pred] for pred in predictions]
        return {"predictions": mapped_predictions}  # Zwrócenie listy z nazwami
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
