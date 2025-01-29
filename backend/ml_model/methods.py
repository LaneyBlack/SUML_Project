"""
This module contains utility functions and classes for text classification,
fine-tuning a ml_model, generating attention maps, and plotting training history.
"""
import os
import io
import shutil
import seaborn
import matplotlib.pyplot as plt
import torch
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments
)
from backend.models.prediction import Label


MODEL_PATH = "ml_model/complete_model"

if not os.path.exists(MODEL_PATH):
    # Create a directory and download the model
    os.makedirs(MODEL_PATH)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained("ml_model/complete_model")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.save_pretrained("ml_model/complete_model")
else:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)


def predict_text(title: str, text: str):
    """
    Predict the label for the given title and text using the trained ml_model.
    Args:
        title (str): The title of the text.
        text (str): The main content of the text.
    Returns:
        dict: A dictionary containing the predicted label and confidence score.
    """
    model.eval()
    # Combine title and text for input
    input_text = f"[TITLE] {title} [TEXT] {text}"

    # Tokenize and predict probabilities
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()

    # Predicted label
    predicted_label = torch.argmax(logits, dim=1).item()

    # Match label with its name for displaying
    label_mapping = {0: Label.REAL, 1: Label.FAKE}
    label = label_mapping[predicted_label].name
    # Confidence score for the predicted label
    confidence = probabilities[predicted_label] * 100

    return {
        "label": label,
        "confidence": round(confidence, 2)  # Confidence as a percentage
    }


def generate_attention_map(text, output_path="charts/attention_map.png", max_tokens=10):
    """
    Generate an attention map for the given text.
    Args:
        text (str): The input text for which the attention map will be generated.
        output_path (str, optional):
            Path to save the generated attention map image.
            Defaults to "charts/attention_map.png".
        max_tokens (int, optional):
            Maximum number of tokens to display in the attention map. Defaults to 10.
    Returns:
        io.BytesIO: A buffer containing the generated attention map image.
    """
    # Tokenize the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Get attention weights from the ml_model
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # List of attention weights

    # Choose the attention weights from the last layer
    attention_weights = attentions[-1].squeeze(0).mean(0).numpy()

    # Get the tokenized words
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).numpy())

    # Limit the number of tokens and attention weights
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        attention_weights = attention_weights[:max_tokens, :max_tokens]

    # Plot the attention map
    plt.figure(figsize=(12, 8))
    seaborn.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="coolwarm",
        annot=False,
        cbar=True
    )

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(f"Attention Map (Limited to Top {max_tokens} Tokens)")
    plt.xlabel("Input Tokens")
    plt.ylabel("Input Tokens")
    plt.tight_layout()

    # Save to a file
    plt.savefig(output_path)

    # Save to an in-memory bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()

    return buffer


def calculate_loss(text, label):
    """
    Calculate the loss for a given text and label.
    Args:
        text (str): The input text for loss calculation.
        label (int): The label associated with the text (e.g., 0 or 1).
    Returns:
        float: The calculated loss.
    """
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding="max_length",
        max_length=128)
    with torch.no_grad():
        outputs = model(**inputs, labels=torch.tensor([label]))
        return outputs.loss.item()


class FineTuneDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for fine-tuning.
    Attributes:
        inputs (dict): Tokenized input data.
        labels (list): List of labels for the data.
    """

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def fine_tune_model(model_path: str, title: str, text: str, label: Label):
    """
    Fine-tune the ml_model using the given text and label.
    Args:
        model_path (str): Path to the trained ml_model.
        title (str): The title of the text.
        text (str): The main content of the text.
        label (Label): The label associated with the text ("REAL" or "FAKE").
            value (int): The value of this Enum
            name (string): String associated with this Enum state
    Returns:
        dict:
            A dictionary containing the fine-tuning message & loss values.
    """
    try:

        # Preparing the data
        input_text = f"[TITLE] {title} [TEXT] {text}"
        label_id = label.value

        # Caclulate loss before training
        loss_before = calculate_loss(input_text, label_id)

        # Tokenizing data for train
        inputs = tokenizer(
            [input_text],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        # Creating PyTorch dataset

        dataset = FineTuneDataset(inputs, [label_id])

        training_args = TrainingArguments(
            output_dir="fine_tune_output",
            num_train_epochs=1,  # Krótki trening dla douczania
            per_device_train_batch_size=1,
            logging_dir="fine_tune_logs",
            logging_steps=1,
            save_strategy="no"  # Nie zapisujemy modelu po każdej epoce
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )

        trainer.train()
        # Calculate loss after training
        loss_after = calculate_loss(input_text, label_id)

        # Temp folder to save the ml_model
        temp_model_path = f"{model_path}_temp"
        if os.path.exists(temp_model_path):
            shutil.rmtree(temp_model_path)  # Remove the temp folder if it exists

        # Saving pretrained temp ml_model
        model.save_pretrained(temp_model_path)
        tokenizer.save_pretrained(temp_model_path)
        # save_model(temp_model_path)

        # Move to original catalog
        if os.path.exists(model_path):
            shutil.rmtree(model_path)  # Remove the old one
        shutil.move(temp_model_path, model_path)

        return {
            "message": "Model successfully fine-tuned.",
            "loss_before": loss_before,
            "loss_after": loss_after
        }

    except (OSError, RuntimeError) as e:
        print(f"Error during fine-tuning: {str(e)}")
    return {
        "message": "Fine-tuning failed.",
        "error": str(e),
        "loss_before": None,
        "loss_after": None
    }


# def save_model(temp_model_path):
#     """
#     Save the ml_model and tokenizer to the specified temporary path.
#     Args:
#         temp_model_path (str):
#             The temporary directory path where the ml_model and tokenizer will be saved.
#     Returns:
#         None
#     """
#


def plot_training_history(train_accuracies, val_accuracies, output_path="training_accuracy.png"):
    """
    Generate a training and validation accuracy chart.
    Args:
        train_accuracies (list): List of training accuracies per epoch.
        val_accuracies (list): List of validation accuracies per epoch.
        output_path (str, optional):
            Path to save the generated chart. Defaults to "training_accuracy.png".
    Returns:
        io.BytesIO: A buffer containing the generated chart image.
    Raises:
        RuntimeError: If an error occurs during chart generation.
    """
    try:
        epochs = range(1, len(train_accuracies) + 1)  # Epoch indices
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_accuracies, 'r-', label='train')  # Red line for training accuracy
        plt.plot(epochs, val_accuracies, 'b-', label='val')  # Blue line for validation accuracy
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        # Save to an in-memory bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()

        return buffer
    except RuntimeError as e:
        raise RuntimeError(f"Failed to plot training history: {e}") from e
