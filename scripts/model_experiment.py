from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = Path("./data/clean/clean_data.csv")
OUTPUT_PATH = Path("./outputs/experiment_results.txt")

def print_classification_reports():
    """Prints the classification reports to the terminal."""
    print("Baseline Model Results (C value 1.0): ")
    print(baseline_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy * 100}%.\n")

    print("-" * 100)

    print("Adjusted C value (0.1) Results: ")
    print(adjust_1_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_1_predictions))
    print(f"Accuracy: {adjust_1_accuracy * 100}%.")

    print("-" * 100)
    
    print("Adjusted C value (0.5) Results: ")
    print(adjust_2_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_2_predictions))
    print(f"Accuracy: {adjust_2_accuracy * 100}%.")

    print("-" * 100)
        
    print("Adjusted C value (2.0) Results: ")
    print(adjust_3_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_3_predictions))
    print(f"Accuracy: {adjust_3_accuracy * 100}%.")

    print("-" * 100)
            
    print("Adjusted ngram range (1, 2) Results: ")
    print(adjust_4_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_4_predictions))
    print(f"Accuracy: {adjust_4_accuracy * 100}%.")

    print("-" * 100)
                
    print("Adjusted ngram range (1, 3) Results: ")
    print(adjust_5_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_5_predictions))
    print(f"Accuracy: {adjust_5_accuracy * 100}%.")

    print("-" * 100)

    print("sublinear_tf (true) Results: ")
    print(adjust_6_results)
    print(f"\nClassification Report:")
    print(classification_report(y_test, adjust_6_predictions))
    print(f"Accuracy: {adjust_6_accuracy * 100}%.")

    print("-" * 100)

    print("\nCorrect predictions:")
    print("Baseline:", (predictions == y_test).sum())
    print("C=0.1:", (adjust_1_predictions == y_test).sum())
    print("C=0.5:", (adjust_2_predictions == y_test).sum())
    print("C=2.0:", (adjust_3_predictions == y_test).sum())
    print("ngram = 1,2", (adjust_4_predictions == y_test).sum())
    print("ngram = 1,3", (adjust_5_predictions == y_test).sum())
    print("sublinear_tf = True", (adjust_6_predictions == y_test).sum())

    print("C=0.1 identical:", (predictions == adjust_1_predictions).all())
    print("C=0.5 identical:", (predictions == adjust_2_predictions).all())
    print("C=2.0 identical:", (predictions == adjust_3_predictions).all())
    print("ngram = 1,2 identical:", (predictions == adjust_4_predictions).all())
    print("ngram = 1,3 identical:", (predictions == adjust_5_predictions).all())
    print("sublinear_tf identical:", (predictions == adjust_6_predictions).all())

def save_experiment_results():
    """Saves the model experiment results to a text file."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:

        file.write("MODEL EXPERIMENTATION RESULTS\n")
        file.write("=" * 100 + "\n\n")

        experiments = [
            ("Baseline (C=1.0)", baseline_results, predictions,
             accuracy, baseline_scores),
            ("C=0.1", adjust_1_results, adjust_1_predictions,
             adjust_1_accuracy, exp_1_scores),
            ("C=0.5", adjust_2_results, adjust_2_predictions,
             adjust_2_accuracy, exp_2_scores),
            ("C=2.0", adjust_3_results, adjust_3_predictions,
             adjust_3_accuracy, exp_3_scores),
            ("ngram_range=(1, 2)", adjust_4_results, adjust_4_predictions,
             adjust_4_accuracy, exp_4_scores),
            ("ngram_range=(1, 3)", adjust_5_results, adjust_5_predictions,
             adjust_5_accuracy, exp_5_scores),
            ("sublinear_tf=True", adjust_6_results, adjust_6_predictions,
             adjust_6_accuracy, exp_6_scores)
        ]

        for name, results, predicted, test_accuracy, cv_scores in experiments:

            file.write(f"{name} RESULTS\n")
            file.write("-" * 100 + "\n")

            file.write("Classification results:\n")
            file.write(results.to_string(index=False) + "\n\n")

            file.write("Classification report:\n")
            file.write(
                classification_report(y_test, predicted, zero_division=0)
                + "\n"
            )

            file.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")

            file.write("Cross-validation results:\n")
            file.write(
                f"Individual fold accuracies: {cv_scores['test_accuracy']}\n"
            )
            file.write(
                f"Individual fold F1 scores: {cv_scores['test_f1_macro']}\n"
            )
            file.write(
                f"Mean accuracy: {cv_scores['test_accuracy'].mean() * 100:.2f}%\n"
            )
            file.write(
                f"Mean F1: {cv_scores['test_f1_macro'].mean() * 100:.2f}%\n"
            )
            file.write(
                f"Accuracy standard deviation: "
                f"{cv_scores['test_accuracy'].std():.4f}\n"
            )
            file.write(
                f"F1 standard deviation: "
                f"{cv_scores['test_f1_macro'].std():.4f}\n\n"
            )

            file.write("=" * 100 + "\n\n")

        file.write("COMPARISON WITH BASELINE\n")
        file.write("-" * 100 + "\n")

        for name, _, predicted, _, _ in experiments:
            correct = (predicted == y_test).sum()
            identical = (predictions == predicted).all()

            file.write(
                f"{name}: {correct}/{len(y_test)} correct predictions\n"
            )
            file.write(
                f"Identical predictions to baseline: {identical}\n\n"
            )

    print(f"Experiment results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    X = df["request_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline Logistic Regression model
    lr_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression())])
    lr_pipeline.fit(X_train, y_train)

    # Baseline cross-validation
    baseline_scores = cross_validate(lr_pipeline, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])

    print("Baseline Individual fold accuracies:", baseline_scores["test_accuracy"])
    print("Baseline Individual fold F1 scores:", baseline_scores["test_f1_macro"])
    print("Baseline Mean accuracy:", baseline_scores["test_accuracy"].mean())
    print("Baseline Mean F1:", baseline_scores["test_f1_macro"].mean())
    print("Baseline accuracy std:", baseline_scores["test_accuracy"].std())
    print("Baseline F1 std:", baseline_scores["test_f1_macro"].std())

    # Logistic Regression model with C value of 0.1
    lr_adjust_1 = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(C=0.1))])
    lr_adjust_1.fit(X_train, y_train)

    # Logistic Regression model with C value of 0.1 cross-validation
    exp_1_scores = cross_validate(lr_adjust_1, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])
    
    print("EXP 1 (C=0.1) Individual fold accuracies:", exp_1_scores["test_accuracy"])
    print("EXP 1 (C=0.1) Individual fold F1 scores:", exp_1_scores["test_f1_macro"])
    print("EXP 1 (C=0.1) Mean accuracy:", exp_1_scores["test_accuracy"].mean())
    print("EXP 1 (C=0.1) Mean F1:", exp_1_scores["test_f1_macro"].mean())
    print("EXP 1 (C=0.1) accuracy std:", exp_1_scores["test_accuracy"].std())
    print("EXP 1 (C=0.1) F1 std:", exp_1_scores["test_f1_macro"].std())

    # Logistic Regression model with C value of 0.5
    lr_adjust_2 = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(C=0.5))])
    lr_adjust_2.fit(X_train, y_train)

    # Logistic Regression model with C value of 0.5 cross-validation
    exp_2_scores = cross_validate(lr_adjust_2, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])
    
    print("EXP 2 (C=0.5) Individual fold accuracies:", exp_2_scores["test_accuracy"])
    print("EXP 2 (C=0.5) Individual fold F1 scores:", exp_2_scores["test_f1_macro"])
    print("EXP 2 (C=0.5) Mean accuracy:", exp_2_scores["test_accuracy"].mean())
    print("EXP 2 (C=0.5) Mean F1:", exp_2_scores["test_f1_macro"].mean())
    print("EXP 2 (C=0.5) accuracy std:", exp_2_scores["test_accuracy"].std())
    print("EXP 2 (C=0.5) F1 std:", exp_2_scores["test_f1_macro"].std())

    # Logistic Regression model with C value of 2.0
    lr_adjust_3 = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(C=2.0))])
    lr_adjust_3.fit(X_train, y_train)

    # Logistic Regression model with C value of 2.0 cross-validation  
    exp_3_scores = cross_validate(lr_adjust_3, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])
    
    print("EXP 3 (C=2.0) Individual fold accuracies:", exp_3_scores["test_accuracy"])
    print("EXP 3 (C=2.0) Individual fold F1 scores:", exp_3_scores["test_f1_macro"])
    print("EXP 3 (C=2.0) Mean accuracy:", exp_3_scores["test_accuracy"].mean())
    print("EXP 3 (C=2.0) Mean F1:", exp_3_scores["test_f1_macro"].mean())
    print("EXP 3 (C=2.0) accuracy std:", exp_3_scores["test_accuracy"].std())
    print("EXP 3 (C=2.0) F1 std:", exp_3_scores["test_f1_macro"].std())

    # Logistic Regression model with Adjusted TF-IDF Vectorizer to have ngram_range of 1, 2
    lr_adjust_4 = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))), ('lr', LogisticRegression())])
    lr_adjust_4.fit(X_train, y_train)

    # Logistic Regression model with Adjusted TF-IDF Vectorizer to have ngram_range of 1, 2 cross-validation          
    exp_4_scores = cross_validate(lr_adjust_4, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])
    
    print("EXP 4 (ngram_range=1,2) Individual fold accuracies:", exp_4_scores["test_accuracy"])
    print("EXP 4 (ngram_range=1,2) Individual fold F1 scores:", exp_4_scores["test_f1_macro"])
    print("EXP 4 (ngram_range=1,2) Mean accuracy:", exp_4_scores["test_accuracy"].mean())
    print("EXP 4 (ngram_range=1,2) Mean F1:", exp_4_scores["test_f1_macro"].mean())
    print("EXP 4 (ngram_range=1,2) accuracy std:", exp_4_scores["test_accuracy"].std())
    print("EXP 4 (ngram_range=1,2) F1 std:", exp_4_scores["test_f1_macro"].std())

    # Logistic Regression model with Adjusted TF-IDF Vectorizer to have ngram_range 1, 3
    lr_adjust_5 = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3))), ('lr', LogisticRegression())])
    lr_adjust_5.fit(X_train, y_train)

    # Logistic Regression model with Adjusted TF-IDF Vectorizer to have ngram_range of 1, 3 cross-validation
    exp_5_scores = cross_validate(lr_adjust_5, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])
    
    print("EXP 5 (ngram_range=1,3) Individual fold accuracies:", exp_5_scores["test_accuracy"])
    print("EXP 5 (ngram_range=1,3) Individual fold F1 scores:", exp_5_scores["test_f1_macro"])
    print("EXP 5 (ngram_range=1,3) Mean accuracy:", exp_5_scores["test_accuracy"].mean())
    print("EXP 5 (ngram_range=1,3) Mean F1:", exp_5_scores["test_f1_macro"].mean())
    print("EXP 5 (ngram_range=1,3) accuracy std:", exp_5_scores["test_accuracy"].std())
    print("EXP 5 (ngram_range=1,3) F1 std:", exp_5_scores["test_f1_macro"].std())

    # Logistic Regress model with Adjusted TF-IDF Vectorizer to have sublinear term frequencies
    lr_adjust_6 = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True)), ('lr', LogisticRegression())])
    lr_adjust_6.fit(X_train, y_train)

    # Logistic Regression model with Adjusted TF-IDF Vectorizer to have sublinear term frequencies
    exp_6_scores = cross_validate(lr_adjust_6, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"])
    
    print("EXP 6 (sublinear_tf=True) Individual fold accuracies:", exp_6_scores["test_accuracy"])
    print("EXP 6 (sublinear_tf=True) Individual fold F1 scores:", exp_6_scores["test_f1_macro"])
    print("EXP 6 (sublinear_tf=True) Mean accuracy:", exp_6_scores["test_accuracy"].mean())
    print("EXP 6 (sublinear_tf=True) Mean F1:", exp_6_scores["test_f1_macro"].mean())
    print("EXP 6 (sublinear_tf=True) accuracy std:", exp_6_scores["test_accuracy"].std())
    print("EXP 6 (sublinear_tf=True) F1 std:", exp_6_scores["test_f1_macro"].std())
    
    predictions = lr_pipeline.predict(X_test)
    adjust_1_predictions = lr_adjust_1.predict(X_test)
    adjust_2_predictions = lr_adjust_2.predict(X_test)
    adjust_3_predictions = lr_adjust_3.predict(X_test)
    adjust_4_predictions = lr_adjust_4.predict(X_test)
    adjust_5_predictions = lr_adjust_5.predict(X_test)
    adjust_6_predictions = lr_adjust_6.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    adjust_1_accuracy = accuracy_score(y_test, adjust_1_predictions)
    adjust_2_accuracy = accuracy_score(y_test, adjust_2_predictions)
    adjust_3_accuracy = accuracy_score(y_test, adjust_3_predictions)
    adjust_4_accuracy = accuracy_score(y_test, adjust_4_predictions)
    adjust_5_accuracy = accuracy_score(y_test, adjust_5_predictions)
    adjust_6_accuracy = accuracy_score(y_test, adjust_6_predictions)
    
    baseline_results = pd.DataFrame({
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

    adjust_6_results = pd.DataFrame({
                        "request_text": X_test,
                        "actual_category": y_test,
                        "predicted_category": adjust_6_predictions
                    })

    save_experiment_results()