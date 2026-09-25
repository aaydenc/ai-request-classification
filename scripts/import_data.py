import pandas as pd

new_file_path = "./data/clean/clean_data.csv"


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
    dataframe["request_text"] = dataframe["request_text"].str.strip().str.lower()
    dataframe["suburb"] = dataframe["suburb"].str.strip().str.lower()

    return dataframe


if __name__ == "__main__":
    df = pd.read_csv("./data/raw/raw_data.csv")

    copy_df = df.copy()
    
    clean_df = clean_data(copy_df)
    clean_df = check_duplicate_data(clean_df)

    print(clean_df)

    clean_df.to_csv(new_file_path, index=False)