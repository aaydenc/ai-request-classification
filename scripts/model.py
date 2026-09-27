import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data_path = "./data/clean/clean_data.csv"

if __name__ == "__main__":
    df = pd.read_csv(data_path)

    X = df['request_text']
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer()

    model = LogisticRegression()

    tf_pipeline = Pipeline([('tfidf', vectorizer), ('lr', model)])
    
    tf_pipeline.fit(X_train, y_train)

    predictions = tf_pipeline.predict(X_test)

    results = pd.DataFrame({
    "actual_catergory": y_test,
    "predicted_catergory": predictions
    })

    print(results)
