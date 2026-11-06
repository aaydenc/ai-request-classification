import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

DATA_PATH = "./data/clean/clean_data.csv"

def print_classification_reports():
    """Prints the classification reports to the terminal."""
    print("Predicted Results: ")
    print(results.head())
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy * 100}%.\n")

    print("-" * 100)

    print("Adjusted C (0.1) Results: ")
    print(adjust_1_results.head())
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_1_predictions))
    print(f"Accuracy: {adjust_1_predictions_accr * 100}%.")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X = df["request_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # Baseline Logistic Regression model
    lr_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression())])
    lr_pipeline.fit(X_train, y_train)

    # Logistic Regression model with 0.1 C Value
    lr_adjust_1= Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression())])
    lr_adjust_1.fit(X_train, y_train)
    
    predictions = lr_pipeline.predict(X_test)
    adjust_1_predictions = lr_adjust_1.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    adjust_1_predictions_accr = accuracy_score(y_test, adjust_1_predictions)
    
    results = pd.DataFrame({
        "request_text": X_test,
        "actual_category": y_test,
        "predicted_category": predictions
    })

    adjust_1_results = pd.DataFrame({
        "request_text": X_test,
        "actual_category": y_test,
        "predicted_category": adjust_1_predictions
    })

    print_classification_reports()