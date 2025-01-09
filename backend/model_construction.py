from backend.import_requirements import np, pd, train_test_split, TfidfVectorizer, tqdm, vstack, joblib, LinearSVC
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import os

DATA_DIR = "data/dataset.csv"
COMPLETE_MODEL_DIR = "models/complete_model.joblib"
FEATURE_WEIGHTS_HEATMAP_DIR = "charts/feature_weights_heatmap.png"
IMPORTANCE_HEATMAP_DIR = "charts/importance_heatmap.png"


def organize_data(model_data_path):
    model_data = pd.read_csv(model_data_path)
    model_data['fake'] = model_data['label'].apply(lambda x: 0 if x == "REAL" else 1)
    model_data = model_data.drop("label", axis=1)
    return model_data


def train_data(data):
    # Przygotowanie danych wejściowych
    x = data[['title', 'text']]  # Użycie dwóch kolumn
    y = data['fake']  # Kolumna docelowa

    # Podział danych na treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Wektoryzatory dla `title` i `text`
    # vectorizer_title = TfidfVectorizer(stop_words="english", max_df=0.7)
    # vectorizer_text = TfidfVectorizer(stop_words="english", max_df=0.7)
    """
    ngram_range = (1, 2):
    It includes both single words(unigrams) and pairs of consecutive words(bigrams).
    The feature matrix will include additional columns representing bigrams(e.g., "fake news," "breaking story").
    """
    vectorizer_title = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2))
    vectorizer_text = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2))

    # Wektoryzacja w partiach dla `title`
    title_train_chunks = [X_train['title'].iloc[i:i + 100] for i in range(0, len(X_train), 100)]
    title_train_vectorized_chunks = []
    for chunk in tqdm(title_train_chunks, desc="Vectorizing Title Data"):
        if len(title_train_vectorized_chunks) == 0:  # Pierwszy fragment
            vectorizer_title.fit(chunk)
        title_train_vectorized_chunks.append(vectorizer_title.transform(chunk))
    title_train_vectorized = vstack(title_train_vectorized_chunks)

    # Wektoryzacja w partiach dla `text`
    text_train_chunks = [X_train['text'].iloc[i:i + 100] for i in range(0, len(X_train), 100)]
    text_train_vectorized_chunks = []
    for chunk in tqdm(text_train_chunks, desc="Vectorizing Text Data"):
        if len(text_train_vectorized_chunks) == 0:  # Pierwszy fragment
            vectorizer_text.fit(chunk)
        text_train_vectorized_chunks.append(vectorizer_text.transform(chunk))
    text_train_vectorized = vstack(text_train_vectorized_chunks)

    # Połączenie wektorów `title` i `text`
    X_train_vectorized = hstack([title_train_vectorized, text_train_vectorized])

    # Wektoryzacja danych testowych
    title_test_vectorized = vectorizer_title.transform(X_test['title'])
    text_test_vectorized = vectorizer_text.transform(X_test['text'])
    X_test_vectorized = hstack([title_test_vectorized, text_test_vectorized])

    # Trenowanie modelu
    model = LinearSVC()
    model.fit(X_train_vectorized, y_train)

    # Zwrócenie modelu, wektoryzatorów i danych testowych
    return model, (vectorizer_title, vectorizer_text), X_test_vectorized, y_test


def feature_weights_heatmap(vectorizer_title, vectorizer_text, model, output_path=FEATURE_WEIGHTS_HEATMAP_DIR,
                            top_n=30):
    # Pobieranie cech i wag
    title_features = vectorizer_title.get_feature_names_out()
    text_features = vectorizer_text.get_feature_names_out()
    model_weights = model.coef_[0]

    # Połącz cechy i wagi
    all_features = list(title_features) + list(text_features)
    all_weights = model_weights

    # Tworzenie DataFrame
    feature_weights = pd.DataFrame({
        "Feature": all_features,
        "Weight": all_weights
    })

    # Posortuj według znaczenia wag
    feature_weights["AbsWeight"] = feature_weights["Weight"].abs()
    feature_weights = feature_weights.sort_values(by="AbsWeight", ascending=False)

    # Wybierz tylko top N cech
    top_features = feature_weights.head(top_n)

    # Tworzenie heatmapy
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        top_features[["Weight"]].T,
        cmap="coolwarm",
        annot=True,
        cbar=True,
        xticklabels=top_features["Feature"],
        yticklabels=["Waga"]
    )
    plt.title("Top {} wag cech dla tytułu i tekstu".format(top_n))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compare_title_text_importance(vectorizer_title, vectorizer_text, model, output_path=IMPORTANCE_HEATMAP_DIR):
    # Pobieranie wag modelu
    model_weights = model.coef_[0]

    # Liczba cech dla tytułów i tekstów
    title_feature_count = len(vectorizer_title.get_feature_names_out())
    text_feature_count = len(vectorizer_text.get_feature_names_out())

    # Wagi dla tytułu i tekstu
    title_weights = model_weights[:title_feature_count]
    text_weights = model_weights[title_feature_count:]

    # Suma wartości bezwzględnych wag
    title_importance = np.sum(np.abs(title_weights))
    text_importance = np.sum(np.abs(text_weights))

    # Dane do wykresu
    labels = ["Title", "Text"]
    values = [title_importance, text_importance]
    colors = ["#1f77b4", "#ff7f0e"]

    # Tworzenie wykresu słupkowego
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=colors)
    plt.title("Znaczenie tytułów i tekstów")
    plt.ylabel("Łączna wartość wag")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_model(clf, X_test_vectorized, y_test, heatmap_path=FEATURE_WEIGHTS_HEATMAP_DIR):
    y_pred = clf.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "heatmap_path": heatmap_path
    }


def save_model(clf, vectorizers, COMPLETE_MODEL_DIR):
    vectorizer_title, vectorizer_text = vectorizers
    model_data = {
        "model": clf,
        "vectorizer_title": vectorizer_title,
        "vectorizer_text": vectorizer_text
    }
    joblib.dump(model_data, COMPLETE_MODEL_DIR)
    print(f"Model and vectorizers saved to {COMPLETE_MODEL_DIR}")
