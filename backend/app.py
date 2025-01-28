"""
This module provides a FastAPI-based backend for ml_model operations,
including generating attention maps, predictions, training charts, and fine-tuning.
"""

import json
import logging
import os
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, StreamingResponse
# Relative imports
from ml_model.construction import (construct)
from ml_model.methods import (generate_attention_map, predict_text, fine_tune_model, plot_training_history)
from models.prediction import Prediction

# Define paths
DATA_DIR = "data/dataset.csv"
MODEL_DIR = "ml_model/complete_model"
CHARTS_DIR = "log/charts"  # Directory where charts will be saved
MODEL_LOG = "log/model_log.json"
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all incoming backend requests and their corresponding responses.
    This middleware logs the HTTP method, URL, status code and the client IP address for every incoming request,
    helping to track and monitor API activity for debugging and auditing purposes.
    Parameters:
        request (Request): The request object containing details such as headers, body, method, and client.
        call_next (Callable): A callable function that is used to generate the response for the request.
    Returns:
        Response: The HTTP response returned to the client after processing the request.
    Logs:
        Logs an entry with the status code, method, URL, and client IP address for each request.
    """
    response = await call_next(request)
    logger.info(f"Incoming request: {response.status_code} {request.method} {request.url} from {request.client.host}")
    return response


@app.get("/", tags=["intro"])
async def index():
    return {"message": "Welcome to the Fake News Detector"}


@app.get("/logs", tags=['info'], response_class=PlainTextResponse)
def get_logs(limit: Optional[int] = 1000):
    """
    Retrieve the backend logs.
    Args:
        limit (int, optional): Number of lines to fetch from the log file. Default is 1000.
    Returns:
        str: The content of the log file or an error message.
    """
    try:
        # Open the log file
        with open(BACKEND_LOG, "r") as file:
            # Read the last `limit` number of lines (to avoid loading too much data)
            logs = file.readlines()
            logs = logs[-limit:]  # Get the last `limit` lines
        # Return logs as a single string
        return "".join(logs)
    except FileNotFoundError:
        # Return a custom error message if the file is not found
        return "Log file not found. Please check the file path."
    except Exception as e:
        # Return a generic error message in case of any other exception
        return f"An error occurred: {str(e)}"


@app.delete("/logs", tags=['info'], response_class=PlainTextResponse)
def clear_logs():
    """
    Clear the backend log file.
    Returns:
        str: A confirmation message or an error message.
    """
    try:
        with open(BACKEND_LOG, "w") as file:
            file.truncate(0)  # Clear the file content
        return "Log file cleared successfully."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing log file: {str(e)}")


@app.get("/construct", tags=['model'])
def construct_model():
    """
    Construct a ml_model from scratch endpoint
    """
    if not os.path.exists(DATA_DIR):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        results = construct()
        return {"message": "Model trained and saved successfully.", "results": results}
    except Exception as e:
        print(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate-chart", tags=['info'])
async def generate_chart():
    """
    Generate a training and validation accuracy chart from saved log history.
    """
    chart_path = f"{CHARTS_DIR}/training_accuracy_chart.png"

    try:
        # Check if log history exists
        if not os.path.exists(MODEL_LOG):
            raise HTTPException(status_code=404, detail="Log history file not found. Train the ml_model first.")

        # Load the log history
        with open(MODEL_LOG, "r") as f:
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

        chart_path = f"{CHARTS_DIR}/training_accuracy_chart.png"
        # Generate the chart as an in-memory bytes buffer
        buffer = plot_training_history(train_accuracies, val_accuracies, output_path=chart_path)

        # Stream the chart back to the client
        return StreamingResponse(buffer,
                                 media_type="image/png",
                                 headers={"Content-Disposition": "inline; filename=training_accuracy_chart.png"}
                                 )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}") from e


@app.post("/attention-map", tags=['info'])
async def get_attention_map(text: str):
    """
    Generate an attention map for the given text.
    """
    if not os.path.exists(MODEL_DIR):
        raise HTTPException(status_code=404, detail="Trained ml_model not found.")
    try:
        output_path = f"{CHARTS_DIR}/attention_map.png"
        generate_attention_map(text=text, output_path=output_path)
        buffer = generate_attention_map(text=text)

        # Stream the image back to the client
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=attention_map.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during fine-tuning: {str(e)}") from e


@app.post("/predict", tags=['model'], status_code=200)
async def get_prediction(request: Prediction):
    """
        Predict the label for the given title and text using the trained ml_model.
        Args:
            request(Prediction): Custom class for representing prediction
                title (str): The title of the text.
                text (str): The main content of the text.
        Returns:
            dict: A dictionary containing the prediction label and confidence score.
    """
    try:
        prediction = predict_text(request.title, request.text)
        return {
            "message": "Prediction successful.",
            "prediction": prediction
        }
    except (FileNotFoundError, RuntimeError) as e:
        return HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.post("/fine-tune", tags=['model'])
async def fine_tune_model(request: Prediction):
    """
    Fine-tune the trained ml_model using the given title, text, and label.

    Args:
        request (Prediction):
            title (str): The title of the text.
            text (str): The main content of the text.
            label (Label): The label for fine-tuning (REAL or FAKE).
    Returns:
        dict: A dictionary containing the fine-tuning results, including the loss before and after.
    """
    if not os.path.exists(MODEL_DIR):
        raise HTTPException(status_code=404, detail="Trained ml_model not found.")
    try:
        results = fine_tune_model(model_path=MODEL_DIR, title=request.title, text=request.text,
                                  label=request.label.value)
        return {
            "message": results["message"],
            "loss_before": results.get("loss_before", "N/A"),
            "loss_after": results.get("loss_after", "N/A")
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during fine-tuning: {str(e)}"
        ) from e


if __name__ == "__main__":
    # 0.0.0.0 and 10_000 to run in Cloud
    uvicorn.run(app, host="0.0.0.0", port=10_000)
