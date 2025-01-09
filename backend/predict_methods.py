from backend.import_requirements import joblib

DATA_DIR = "data/dataset.csv"
COMPLETE_MODEL_DIR = "models/complete_model.joblib"
def predict(text):
    # Wczytaj oba elementy z jednego pliku
    loaded_data = joblib.load(COMPLETE_MODEL_DIR)
    model = loaded_data["model"]
    vectorizer = loaded_data["vectorizer"]
    vectorizer_text = vectorizer.transform(text)
    return model.predict(vectorizer_text)

