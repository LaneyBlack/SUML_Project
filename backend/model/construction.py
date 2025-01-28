"""
This module provides model construction with:
functionality for organizing data, training a DistilBERT-based model,
and evaluating its performance for fake news classification.
"""
import os
import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, TrainerCallback, DistilBertConfig
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "../data/dataset.csv"
CHART_EPOCHS = "../charts/training_accuracy.png"
COMPLETE_MODEL_DIR = "complete_model"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Wyłączenie ostrzeżeń o symlinkach
MODEL_LOG = "../log/model_log.json"


# Data Preparation
def organize_data(model_data_path):
    """
      Organize and preprocess the data for the model.
      Args:
          model_data_path (str): Path to the dataset CSV file.
      Returns:
          pandas.DataFrame: Preprocessed DataFrame with combined text and binary labels.
    """
    model_data = pd.read_csv(model_data_path)
    model_data['fake'] = model_data['label'].apply(lambda x: 0 if x == "REAL" else 1)
    model_data = model_data.drop("label", axis=1)
    model_data['combined_text'] = "[TITLE] " + model_data["title"] + " [TEXT] " + model_data["text"]
    return model_data


class FakeNewsDataset(Dataset):
    """
       Custom PyTorch Dataset for handling fake news data.
       Args:
           texts (list): List of text samples.
           labels (list): List of corresponding labels.
           tokenizer (DistilBertTokenizer): Tokenizer instance for encoding text.
           max_length (int): Maximum length for text sequences.
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Get the number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single sample by index.
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: Dictionary containing input IDs, attention mask, and label.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model.
    Args:
        eval_pred (tuple): Tuple containing logits and labels.
    Returns:
        dict: Dictionary containing the accuracy score.
    """
    logits, labels = eval_pred
    # Check logits and labels, to avoid errors
    if isinstance(logits, torch.Tensor):
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    else:
        predictions = logits.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


# Create a custom config with dropout
config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    dropout=0.3,  # This will apply dropout correctly
    attention_dropout=0.3  # Optional: Increases dropout on attention layers
)


def prepare_datasets(data, tokenizer, max_length):
    """
    Prepare training and testing datasets from the provided data.
    This function splits the input data into training and testing sets, tokenizes the text data,
    and creates PyTorch datasets for use in model training and evaluation.
    Args:
        data (pandas.DataFrame): A DataFrame containing the combined text column ("combined_text")
            and binary labels column ("fake").
        tokenizer (transformers.PreTrainedTokenizer):
            A tokenizer instance (e.g., DistilBertTokenizer)
            for encoding text data.
        max_length (int): The maximum length for text sequences after tokenization.
    Returns:
        tuple: A tuple containing two datasets:
            - train_dataset (FakeNewsDataset): The training dataset.
            - test_dataset (FakeNewsDataset): The testing dataset.
    """
    x = data["combined_text"]
    y = data["fake"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    train_dataset = (
        FakeNewsDataset(x_train.tolist(), y_train.tolist(), tokenizer, max_length=max_length))
    test_dataset = (
        FakeNewsDataset(x_test.tolist(), y_test.tolist(), tokenizer, max_length=max_length))
    return train_dataset, test_dataset


def train_model(data):
    """
    Train and evaluate a DistilBERT model on the provided data.
    Args:
        data (pandas.DataFrame): Preprocessed data containing combined text and binary labels.
    Returns:
        dict: Evaluation results containing metrics like accuracy.
    Raises:
        Exception: If training fails.
    """
    try:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            config=config  # Pass the modified config
        )
        train_dataset, test_dataset = prepare_datasets(data, tokenizer, max_length=128)

        training_args = TrainingArguments(
            output_dir="output",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Add the custom callback for training accuracy
        callback = TrainingAccuracyCallback(trainer, train_dataset)
        trainer.add_callback(callback)
        # Training model
        trainer.train()
        # Evaluate model
        results = trainer.evaluate()
        print("Accuracy:", results["eval_accuracy"])

        # Save the model
        model.save_pretrained(COMPLETE_MODEL_DIR)
        tokenizer.save_pretrained(COMPLETE_MODEL_DIR)

        with open(MODEL_LOG, "w") as f:
            json.dump(trainer.state.log_history, f)
        return results
    except Exception as e:
        print(f"Training failed: {e}")
        raise


class TrainingAccuracyCallback(TrainerCallback):
    """
    Custom callback to log training accuracy at the end of each epoch.
    Args:
        trainer (Trainer): Trainer instance for managing training.
        train_dataset (Dataset): Dataset used for training.
    """
    def __init__(self, trainer, train_dataset):
        self.trainer = trainer
        self.train_dataset = train_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Compute and log training accuracy at the end of an epoch.
        Args:
            args: Training arguments.
            state: Training state.
            control: Training control.
        """
        # Use the trainer instance to make predictions
        predictions = self.trainer.predict(self.train_dataset).predictions
        preds = predictions.argmax(axis=1)

        # Collect labels from the dataset
        labels = [self.train_dataset[idx]['label'].item() for idx in range(len(self.train_dataset))]

        # Calculate training accuracy
        train_accuracy = accuracy_score(labels, preds)
        print(f"Training Accuracy (Epoch {state.epoch}): {train_accuracy}")

        # Log the training accuracy
        state.log_history.append({"accuracy": train_accuracy, "epoch": state.epoch})


def construct():
    """
        Main function to organize data and train the model.
        Raises:
            Exception: If the dataset path is invalid or training fails.
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR}")

    try:
        print("---Organising the data---")
        data = organize_data(DATA_DIR)
        print("---Training a model---")
        results = train_model(data)
        print(f"---Model trained and saved successfully.---")
        print(f"Results: {results}")
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    construct()
