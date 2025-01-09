from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import uvicorn

app = FastAPI()
# Import the existing functions from your backend
from model_construction import organize_data, train_data, save_model, evaluate_model, feature_weights_heatmap, \
    compare_title_text_importance
from predict_methods import predict

# Define paths
DATA_DIR = "data/dataset.csv"
COMPLETE_MODEL_DIR = "models/complete_model.joblib"
FEATURE_WEIGHTS_HEATMAP_DIR = "charts/feature_weights_heatmap.png"
IMPORTANCE_HEATMAP_DIR = "charts/importance_heatmap.png"


# Pydantic model for input text
class PredictionRequest(BaseModel):
    text: List[str]  # Accepts a list of text inputs for predictions


from fastapi.responses import FileResponse


@app.post("/train")
def train_model():
    if not os.path.exists(DATA_DIR):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        # Train the model
        data = organize_data(DATA_DIR)
        model, vectorizers, X_test_vectorized, y_test = train_data(data)
        # Ewaluacja modelu
        stats = evaluate_model(model, X_test_vectorized, y_test)
        # Tworzenie heatmapy wag cech
        vectorizer_title = vectorizers[0]
        vectorizer_text = vectorizers[1]
        feature_weights_heatmap(vectorizer_title, vectorizer_text, model)
        compare_title_text_importance(vectorizer_title, vectorizer_text, model)
        # Zapis modelu i wektoryzatorów
        save_model(model, vectorizers, COMPLETE_MODEL_DIR)
        # Zwracanie statystyk i ścieżki do heatmapy
        return {
            "message": "Model trained and saved successfully.",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-heatmap")
def get_feature_heatmap():
    if not os.path.exists(FEATURE_WEIGHTS_HEATMAP_DIR):
        raise HTTPException(status_code=404, detail="Feature heatmap not found.")
    return FileResponse(FEATURE_WEIGHTS_HEATMAP_DIR, media_type="image/png")
@app.get("/importance-heatmap")
def get_feature_heatmap():
    if not os.path.exists(IMPORTANCE_HEATMAP_DIR):
        raise HTTPException(status_code=404, detail="Feature heatmap not found.")
    return FileResponse(IMPORTANCE_HEATMAP_DIR, media_type="image/png")


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
