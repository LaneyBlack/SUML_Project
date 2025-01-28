import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

from transformers import TrainerCallback
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
    model_data = pd.read_csv(model_data_path)
    model_data['fake'] = model_data['label'].apply(lambda x: 0 if x == "REAL" else 1)
    model_data = model_data.drop("label", axis=1)
    model_data['combined_text'] = "[TITLE] " + model_data["title"] + " [TEXT] " + model_data["text"]
    return model_data


# PyTorch Dataset
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
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
    logits, labels = eval_pred
    # Sprawdź typ logits i labels, aby uniknąć błędów
    if isinstance(logits, torch.Tensor):
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    else:
        predictions = logits.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def train_model(data):
    try:
        x = data["combined_text"]
        y = data["fake"]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

        train_dataset = FakeNewsDataset(X_train.tolist(), y_train.tolist(), tokenizer, max_length=128)
        test_dataset = FakeNewsDataset(X_test.tolist(), y_test.tolist(), tokenizer, max_length=128)

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
        # Trening modelu
        trainer.train()
        # Ewaluacja modelu
        results = trainer.evaluate()
        print("Accuracy:", results["eval_accuracy"])

        # Zapis modelu
        model.save_pretrained(COMPLETE_MODEL_DIR)
        tokenizer.save_pretrained(COMPLETE_MODEL_DIR)

        with open(MODEL_LOG, "w") as f:
            json.dump(trainer.state.log_history, f)
        return results
    except Exception as e:
        print(f"Training failed: {e}")
        raise


class TrainingAccuracyCallback(TrainerCallback):
    def __init__(self, trainer, train_dataset):
        self.trainer = trainer
        self.train_dataset = train_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
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
    if not os.path.exists(DATA_DIR):
        raise Exception(f"Data directory does not exist: {DATA_DIR}")
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
