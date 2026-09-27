import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

data_path = "./data/clean/clean_data.csv"

if __name__ == "__main__":
    df = pd.read_csv(data_path)

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

    print(results)

    print(f"Accuracy: {accuracy * 100}%.\n")
    print(f"Classification Report:")
    print(classification_report(y_test, predictions))

    pipleline_filename = "model/request_classifier_pipeline.joblib"
    
    joblib.dump(model, pipleline_filename)

    print(f"Pipeline saved to {pipleline_filename}")
    