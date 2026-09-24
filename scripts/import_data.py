import pandas as pd

new_file_path = "./data/raw/clean_data.csv"

df = pd.read_csv("./data/raw/raw_data.csv")

def check_duplicate_data(dataframe):
    """ Removed duplicate data from the provided dataframe.
        Outputs new data to a new file.

    Args:
        dataframe: The specified dataframe being checked.

    Returns:
        A dataframe with duplicate data corrected.
    """
    new_df = dataframe.copy()
    new_df.to_csv(new_file_path)

    no_duplicate_data = new_df.drop_duplicates(subset=["request_text", "suburb", "date_reported"])

    return no_duplicate_data
    
def clean_data(dataframe):
    """Cleans the request_text column in the dataframe by stripping it an removing excess whitespace.
    
    Args:
        dataframe: The dataframe that is being cleaned.

    Returns:
        A dataframe with cleaned data.
    """
    clean_data = dataframe["request_text"].str.lower()
    print(clean_data)
    # clean_data["request_text"].str.strip()

    return clean_data

if __name__ == "__main__":
    clean_df = check_duplicate_data(df)
    # print(clean_df)
    clean_data(clean_df)