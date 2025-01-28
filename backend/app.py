import json

from backend.import_requirements import os, FastAPI, HTTPException, uvicorn, Enum
from backend.model.methods import (generate_attention_map, predict_text, fine_tune_model, plot_training_history)
from fastapi.responses import StreamingResponse

app = FastAPI()

# Define paths
DATA_DIR = "data/dataset.csv"
MODEL_PATH = "model/complete_model"
CHARTS_DIR = "charts"  # Directory where charts will be saved
MODEL_LOG_DIR = "log/model_log.json"


@app.get("/generateChart")
def generate_chart_endpoint():
    """
    Generate a training and validation accuracy chart and return it as a downloadable file.
    """
    try:
        # Check if log history exists
        if not os.path.exists(MODEL_LOG_DIR):
            raise HTTPException(status_code=404, detail="Log history file not found. Train the model first.")

        # Load the log history
        with open(MODEL_LOG_DIR, "r") as f:
            log_history = json.load(f)

        # Extract training and validation accuracies
        train_accuracies = [log["accuracy"] for log in log_history if "accuracy" in log]
        val_accuracies = [log["eval_accuracy"] for log in log_history if "eval_accuracy" in log]

        # Ensure both lists have the same length
        min_length = min(len(train_accuracies), len(val_accuracies))
        train_accuracies = train_accuracies[:min_length]
        val_accuracies = val_accuracies[:min_length]
        chart_path = f"{CHARTS_DIR}/training_accuracy_chart.png"
        # Generate the chart as an in-memory bytes buffer
        buffer = plot_training_history(train_accuracies, val_accuracies, output_path=chart_path)

        # Stream the chart back to the client
        return StreamingResponse(buffer, media_type="image/png",
                                 headers={"Content-Disposition": "inline; filename=training_accuracy_chart.png"}
                                 )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")


@app.post("/attention-map")
def attention_map_endpoint(text: str):
    """
    Generate an attention map for the given text and return it as a downloadable file.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Trained model not found.")
    try:
        buffer = generate_attention_map(text=text)

        # Stream the image back to the client
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=attention_map.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating attention map: {str(e)}")


@app.post("/predict")
async def predict_endpoint(title: str, text: str):
    try:
        prediction = predict_text(title, text)
        return {
            "message": "Prediction successful.",
            "prediction": prediction
        }
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


# Enum for label
class Label(str, Enum):
    REAL = "REAL"
    FAKE = "FAKE"


@app.post("/fine-tune")
def fine_tune_endpoint(title: str, text: str, label: Label):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Trained model not found.")
    try:
        results = fine_tune_model(model_path=MODEL_PATH, title=title, text=text, label=label.value)
        return {
            "message": results["message"],
            "loss_before": results.get("loss_before", "N/A"),
            "loss_after": results.get("loss_after", "N/A")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during fine-tuning: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
