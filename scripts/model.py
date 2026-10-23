import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "./data/clean/clean_data.csv"

def save_model_to_disk():
    pipleline_filename = "model/request_classifier_pipeline.joblib"
    
    joblib.dump(tf_pipeline, pipleline_filename)

    print(f"Pipeline saved to {pipleline_filename}")

def save_output_to_file():
    with open("./outputs/classification_report.txt", "w") as f:
        print("Predicted Results: ", file=f)
        print(results.head(), file=f)
        print(f"\nClassification Report:" ,file=f)
        print(classification_report(y_test, predictions), file=f)
        print(f"Accuracy: {accuracy * 100}%.", file=f)

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X = df["request_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer()
    model = LogisticRegression()

    tf_pipeline = Pipeline([('tfidf', vectorizer), ('lr', model)])
    tf_pipeline.fit(X_train, y_train)

    predictions = tf_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    
    results = pd.DataFrame({
        "request_text": X_test,
        "actual_catergory": y_test,
        "predicted_catergory": predictions
    })

    print("Predicted Results: ")
    print(results.head())
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy * 100}%.")