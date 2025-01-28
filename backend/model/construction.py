import json

from backend.import_requirements import pd, train_test_split, DistilBertTokenizer, \
    DistilBertForSequenceClassification, Trainer, TrainingArguments, torch, accuracy_score, Dataset
from backend.import_requirements import os

from transformers import TrainerCallback
from sklearn.metrics import accuracy_score

# Paths
DATA_DIR = "../data/dataset.csv"
CHART_EPOCHS = "../charts/training_accuracy.png"
COMPLETE_MODEL_DIR = "complete_model"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Wyłączenie ostrzeżeń o symlinkach
MODEL_LOG_DIR = "../log/model_log.json"


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
    # Sprawdź typ logits i labels, aby uniknąć błędów
    if isinstance(logits, torch.Tensor):
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    else:
        predictions = logits.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


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

        with open(MODEL_LOG_DIR, "w") as f:
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
        raise Exception(f"Data directory does not exist: {DATA_DIR}")

    try:
        data = organize_data(DATA_DIR)
        results = train_model(data)
        print(f"Model trained and saved successfully."
              f"Results: {results}")
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    construct()
