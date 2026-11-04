import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC

DATA_PATH = "./data/clean/clean_data.csv"

def save_model_to_disk():
    """Saves model pipeline to disk."""
    pipleline_filename = "model/request_classifier_pipeline.joblib"
    
    joblib.dump(lr_pipeline, pipleline_filename)

    print(f"Pipeline saved to {pipleline_filename}")

def save_output_to_file():
    """Saves classification reports to a file."""
    with open("./outputs/classification_report.txt", "w") as f:
        print("Predicted Results: ")
        print(results.head())
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions))
        print(f"Accuracy: {accuracy * 100}%.")
        

def print_classification_reports():
    """Prints the classification reports to the terminal."""
    print("Predicted Results: ")
    print(results.head())
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy * 100}%.")

    print("-" * 100)

    print("SVC Predicted Results: ")
    print(svc_results.head())
    print(f"\nClassification Report:")
    print(classification_report(y_test, svc_predictions))
    print(f"Accuracy: {svc_accuracy * 100}%.")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X = df["request_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression()
    svc_model = LinearSVC()

    lr_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('lr', model)])
    lr_pipeline.fit(X_train, y_train)

    svc_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('svc', svc_model)])
    svc_pipeline.fit(X_train, y_train)

    predictions = lr_pipeline.predict(X_test)
    svc_predictions = svc_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    svc_accuracy = accuracy_score(y_test, svc_predictions)
    
    results = pd.DataFrame({
        "request_text": X_test,
        "actual_category": y_test,
        "predicted_category": predictions
    })

    svc_results = pd.DataFrame({
        "request_text": X_test,
        "actual_category": y_test,
        "predicted_category" : svc_predictions
    })

    print_classification_reports()