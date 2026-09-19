import pandas as pd

df = pd.read_csv("./data/raw/raw_data.csv")

def check_duplicate_data(dataframe):
    """ Removed duplicate data from the provided dataframe.

    Args:
        dataframe: The specified dataframe being checked.

    Returns:
        A dataframe with duplicate data corrected.
    """
    no_duplicate_data = dataframe.drop_duplicates(subset=["request_text", "suburb", "date_reported"])

    return no_duplicate_data

def clean_data(dataframe):
    """Cleans the data in the dataframe to make it consistent.
    
    Args:
        dataframe: The dataframe that is being cleaned.

    Returns:
        A dataframe with cleaned data.
    """
    

if __name__ == "__main__":
    # print(df)
    clean_df = check_duplicate_data(df)