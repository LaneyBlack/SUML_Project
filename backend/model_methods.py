from backend.import_requirements import plt, sns, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, \
    TrainingArguments, torch, os


def predict_text(title: str, text: str, model_path: str):
    # Load the model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
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

    # Define custom labels (adjust as per your model training)
    id2label = {0: "REAL", 1: "FAKE"}  # Example mapping
    label_name = id2label.get(predicted_label, f"LABEL_{predicted_label}")

    # Confidence score for the predicted label
    confidence = probabilities[predicted_label] * 100

    return {
        "label_name": label_name,
        "confidence": round(confidence, 2)  # Confidence as a percentage
    }


def generate_attention_map(model_path, text, output_path="charts/attention_map.png", max_tokens=10):
    # Load the tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    # Tokenize the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Get attention weights from the model
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
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="coolwarm",
        annot=False,
        cbar=True
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Attention Map (Limited to Top {} Tokens)".format(max_tokens))
    plt.xlabel("Input Tokens")
    plt.ylabel("Input Tokens")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

import shutil


def fine_tune_model(model_path: str, title: str, text: str, label: str):
    try:
        # Model and tokeniser setup
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)

        # Preparing the data
        input_text = f"[TITLE] {title} [TEXT] {text}"
        label_id = 0 if label == "REAL" else 1

        def calculate_loss(model, tokenizer, text, label):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            with torch.no_grad():
                outputs = model(**inputs, labels=torch.tensor([label]))
                return outputs.loss.item()

        # Caclulate loss before training
        loss_before = calculate_loss(model, tokenizer, input_text, label_id)

        # Tokenizing data for train
        inputs = tokenizer(
            [input_text],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        # Creating PyTorch dataset
        class FineTuneDataset(torch.utils.data.Dataset):
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
        loss_after = calculate_loss(model, tokenizer, input_text, label_id)

        # Temp folder to save the model
        temp_model_path = f"{model_path}_temp"
        if os.path.exists(temp_model_path):
            shutil.rmtree(temp_model_path)  # Remove the temp folder if it exists

        # Saving pretrained temp model
        model.save_pretrained(temp_model_path)
        tokenizer.save_pretrained(temp_model_path)

        # Move to original catalog
        if os.path.exists(model_path):
            shutil.rmtree(model_path)  # Remove the old one
        shutil.move(temp_model_path, model_path)

        return {
            "message": "Model successfully fine-tuned.",
            "loss_before": loss_before,
            "loss_after": loss_after
        }

    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        return {
            "message": "Fine-tuning failed.",
            "error": str(e),
            "loss_before": None,
            "loss_after": None
        }
