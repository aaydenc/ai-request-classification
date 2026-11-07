import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "./data/clean/clean_data.csv"

def print_classification_reports():
    """Prints the classification reports to the terminal."""
    print("Basline Model Results (C value 1.0): ")
    print(results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy * 100}%.\n")

    print("-" * 100)

    print("Adjusted C value (0.1) Results: ")
    print(adjust_1_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_1_predictions))
    print(f"Accuracy: {adjust_1_predictions_accr * 100}%.")

    print("-" * 100)
    
    print("Adjusted C value (0.5) Results: ")
    print(adjust_2_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_2_predictions))
    print(f"Accuracy: {adjust_2_predictions_accr * 100}%.")

    print("-" * 100)
        
    print("Adjusted C value (2.0) Results: ")
    print(adjust_3_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_3_predictions))
    print(f"Accuracy: {adjust_3_predictions_accr * 100}%.")

    print("-" * 100)
            
    print("Adjusted ngram range (1, 2) Results: ")
    print(adjust_4_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_4_predictions))
    print(f"Accuracy: {adjust_4_predictions_accr * 100}%.")

    print("-" * 100)
                
    print("Adjusted ngram range (1, 3) Results: ")
    print(adjust_5_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_5_predictions))
    print(f"Accuracy: {adjust_5_predictions_accr * 100}%.")

    print("-" * 100)

    print("\nCorrect predictions:")
    print("Baseline:", (predictions == y_test).sum())
    print("C=0.1:", (adjust_1_predictions == y_test).sum())
    print("C=0.5:", (adjust_2_predictions == y_test).sum())
    print("C=2.0:", (adjust_3_predictions == y_test).sum())
    print("ngram = 1,2", (adjust_4_predictions == y_test).sum())
    print("ngram = 1,4", (adjust_5_predictions == y_test).sum())

    print("C=0.1 identical:", (predictions == adjust_1_predictions).all())
    print("C=0.5 identical:", (predictions == adjust_2_predictions).all())
    print("C=2.0 identical:", (predictions == adjust_3_predictions).all())
    print("ngram = 1,2 identical:", (predictions == adjust_4_predictions).all())
    print("ngram = 1,3 identical:", (predictions == adjust_5_predictions).all())

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X = df["request_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # Baseline Logistic Regression model
    lr_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression())])
    lr_pipeline.fit(X_train, y_train)

    # Logistic Regression model with C value of 0.1
    lr_adjust_1 = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(C=0.1))])
    lr_adjust_1.fit(X_train, y_train)

    # Logistic Regression model with C value of 0.5
    lr_adjust_2 = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(C=0.5))])
    lr_adjust_2.fit(X_train, y_train)

    # Lofistic Regression model with C value of 2.0
    lr_adjust_3 = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(C=2.0))])
    lr_adjust_3.fit(X_train, y_train)

    # Logistic Regression model with Adjusted TF-IDF Vectorizer to have ngram_range of 1, 2
    lr_adjust_4 = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))), ('lr', LogisticRegression())])
    lr_adjust_4.fit(X_train, y_train)

    # Logistic Regression model with Adjusted TF-IDF Vectorizer to have ngram_range 1, 3
    lr_adjust_5 = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3))), ('lr', LogisticRegression())])
    lr_adjust_5.fit(X_train, y_train)
    
    predictions = lr_pipeline.predict(X_test)
    adjust_1_predictions = lr_adjust_1.predict(X_test)
    adjust_2_predictions = lr_adjust_2.predict(X_test)
    adjust_3_predictions = lr_adjust_3.predict(X_test)
    adjust_4_predictions = lr_adjust_4.predict(X_test)
    adjust_5_predictions = lr_adjust_5.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    adjust_1_predictions_accr = accuracy_score(y_test, adjust_1_predictions)
    adjust_2_predictions_accr = accuracy_score(y_test, adjust_2_predictions)
    adjust_3_predictions_accr = accuracy_score(y_test, adjust_3_predictions)
    adjust_4_predictions_accr = accuracy_score(y_test, adjust_4_predictions)
    adjust_5_predictions_accr = accuracy_score(y_test, adjust_5_predictions)
    
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

    adjust_2_results = pd.DataFrame({
        "request_text": X_test,
        "actual_category": y_test,
        "predicted_category": adjust_2_predictions
    })

    adjust_3_results = pd.DataFrame({
            "request_text": X_test,
            "actual_category": y_test,
            "predicted_category": adjust_3_predictions
        })

    adjust_4_results = pd.DataFrame({
                "request_text": X_test,
                "actual_category": y_test,
                "predicted_category": adjust_4_predictions
            })

    adjust_5_results = pd.DataFrame({
                    "request_text": X_test,
                    "actual_category": y_test,
                    "predicted_category": adjust_5_predictions
                })

    print_classification_reports()