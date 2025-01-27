import json
import logging
import os
import uvicorn
from enum import Enum
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, PlainTextResponse
# Relative imports
from model.construction import (organize_data, train_model)
from model.methods import (generate_attention_map, predict_text, fine_tune_model, plot_training_history)

# Define paths
DATA_DIR = "data/dataset.csv"
COMPLETE_MODEL_DIR = "models/complete_model"
MODEL_PATH = "models/complete_model"
CHARTS_DIR = "charts"  # Directory where charts will be saved
MODEL_LOG_DIR = "log/model_log.json"
BACKEND_LOG = "log/backend.log"

# Setup logger config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BACKEND_LOG),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
# Create a logger instance
logger = logging.getLogger(__name__)
# Application definition
app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    This is a middleware to log all the backend method executions
    @param request: the request object with request body
    @param call_next: The method that should be called to get the response body
    @return: response to the client
    """
    response = await call_next(request)
    logger.info(f"Incoming request: {response.status_code} {request.method} {request.url} from {request.client.host}")
    # logger.info(f"Response status: {response.status_code}")
    return response


@app.get("/logs", response_class=PlainTextResponse)
def get_logs():
    try:
        # Read the log file content
        with open(BACKEND_LOG, "r") as file:
            logs = file.read()
        return logs
    except FileNotFoundError:
        return "Log file not found."


# @app.get("/train")
# def train_model_endpoint():
#     if not os.path.exists(DATA_DIR):
#         raise HTTPException(status_code=404, detail="Dataset not found.")
#     try:
#         data = organize_data(DATA_DIR)
#         results = train_model(data)
#         return {"message": "Model trained and saved successfully.", "results": results}
#     except Exception as e:
#         print(f"Training failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate-chart")
def generate_chart():
    """
    Generate a training and validation accuracy chart from saved log history.
    """
    chart_path = f"{CHARTS_DIR}/training_accuracy_chart.png"

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

        # Generate the plot
        plot_training_history(train_accuracies, val_accuracies, chart_path)

        return {"message": "Chart generated successfully.", "chart_path": chart_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")


@app.post("/attention-map")
def attention_map_endpoint(text: str):
    """
    Generate an attention map for the given text.
    """
    if not os.path.exists(COMPLETE_MODEL_DIR):
        raise HTTPException(status_code=404, detail="Trained model not found.")
    try:
        output_path = f"{CHARTS_DIR}/attention_map.png"
        generate_attention_map(model_path=COMPLETE_MODEL_DIR, text=text, output_path=output_path)
        return {"message": "Attention map generated successfully.", "path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_endpoint(title: str, text: str):
    try:
        prediction = predict_text(title, text, MODEL_PATH)
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
    if not os.path.exists(COMPLETE_MODEL_DIR):
        raise HTTPException(status_code=404, detail="Trained model not found.")
    try:
        results = fine_tune_model(model_path=COMPLETE_MODEL_DIR, title=title, text=text, label=label.value)
        return {
            "message": results["message"],
            "loss_before": results.get("loss_before", "N/A"),
            "loss_after": results.get("loss_after", "N/A")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during fine-tuning: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=10000)
