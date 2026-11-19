from pathlib import Path

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = Path("./data/clean/clean_data.csv")
MODEL_PATH = Path("./model/request_classifier_pipeline.joblib")
OUTPUT_PATH = Path("./outputs/classification_report.txt")

def save_model_to_disk(pipeline):
    """Saves model pipeline to disk."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Pipeline saved to {MODEL_PATH}")

def save_output_to_file(results, y_test, predictions, accuracy):
    """Saves classification reports to a file."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write("Predicted Results:\n")
        file.write(results.head().to_string())
        file.write("\n\nClassification Report:\n")
        file.write(classification_report(y_test, predictions))
        file.write(f"\nAccuracy: {accuracy * 100:.2f}%\n")
        file.write("-" * 100)

    print(f"Classification report saved to {OUTPUT_PATH}")
        

def print_classification_report(results, y_test, predictions, accuracy):
    """Prints the classification reports to the terminal."""
    print("Predicted Results: ")
    print(results.head())
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy * 100}%.")
    print("-" * 100)


def train_model():
    """Load data, train the model and evaluate its performance."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = {"request_text", "category"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must contain request_text and category columns.")

    if df.empty:
        raise ValueError("The dataset contains no records.")

    X = df["request_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logistic_regression_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LogisticRegression())
    ])

    logistic_regression_pipeline.fit(X_train, y_train)

    predictions = logistic_regression_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results = pd.DataFrame({
        "request_text": X_test,
        "actual_category": y_test,
        "predicted_category": predictions
    })

    save_model_to_disk(logistic_regression_pipeline)

    print_classification_report(results, y_test, predictions, accuracy)
    save_output_to_file(results, y_test, predictions, accuracy)

if __name__ == "__main__":
    train_model()