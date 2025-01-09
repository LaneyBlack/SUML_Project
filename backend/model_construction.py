from backend.import_requirements import pd, train_test_split, TfidfVectorizer, tqdm, vstack, joblib, LinearSVC

DATA_DIR = "data/dataset.csv"
COMPLETE_MODEL_DIR = "models/complete_model.joblib"


def organize_data(model_data_path):
    model_data = pd.read_csv(model_data_path)
    model_data['fake'] = model_data['label'].apply(lambda x: 0 if x == "REAL" else 1)
    model_data = model_data.drop("label", axis=1)
    return model_data


def train_data(data):
    x, y = data['text'], data['fake']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    chunk_size = 100  # Adjust based on your dataset size
    X_train_chunks = [X_train[i:i + chunk_size] for i in range(0, len(X_train), chunk_size)]
    vectorizer.fit(X_train)
    # Transform the data chunk by chunk
    X_train_vectorized_chunks = []
    for chunk in tqdm(X_train_chunks, desc="Vectorizing Training Data"):
        X_train_vectorized_chunks.append(vectorizer.transform(chunk))
    X_train_vectorized = vstack(X_train_vectorized_chunks)
    X_test_vectorized = vectorizer.transform(X_test)
    model = LinearSVC()
    model.fit(X_train_vectorized, y_train)
    return model, vectorizer, X_test_vectorized, y_test


def evaluate_model(clf, X_test_vectorized, y_test):
    accuracy = clf.score(X_test_vectorized, y_test)
    print(f"Model accuracy: {accuracy:.2f}")


def save_model(clf, vectorizer, COMPLETE_MODEL_DIR):
    # Zapisz model i vectorizer w jednym pliku
    model_data = {"model": clf, "vectorizer": vectorizer}
    joblib.dump(model_data, COMPLETE_MODEL_DIR)


def predict(text):
    # Wczytaj oba elementy z jednego pliku
    loaded_data = joblib.load(COMPLETE_MODEL_DIR)
    model = loaded_data["model"]
    vectorizer = loaded_data["vectorizer"]
    vectorizer_text = vectorizer.transform(text)
    return model.predict(vectorizer_text)
