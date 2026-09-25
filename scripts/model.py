import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data_path = "./data/clean/clean_data.csv"

if __name__ == "__main__":
    df = pd.read_csv(data_path)

    # vectorizer = TfidfVectorizer()
    # tf_matrix = vectorizer.fit_transform(df["request_text"])

    # print(tf_matrix)