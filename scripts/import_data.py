from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("./data/raw/raw_data.csv")
CLEAN_DATA_PATH = Path("./data/clean/clean_data.csv")

REQUIRED_COLUMNS = {
    "request_id",
    "request_text",
    "category",
    "suburb",
    "priority",
    "date_reported"
}

def check_duplicate_data(dataframe):
    """ Checks provided dataframe for any duplicate records across
        the request_text, suburb, and date_reported columns and
        drops any rows that are duplicates.
        
    Args:
        dataframe: The specified dataframe being checked.

    Returns:
        A dataframe with duplicate data dropped.
    """
    no_duplicate_data = dataframe.drop_duplicates(subset=["request_text", "suburb", "date_reported"])

    return no_duplicate_data


def clean_data(dataframe):
    """Cleans the request_text column to strip excess whitespace and convert to lowercase.
    Checks the request_text column for empty data and replaces them with an empty string.
    Args:
        dataframe: The dataframe that is being cleaned.

    Returns:
        A dataframe with cleaned data.
    """
    cleaned_dataframe = dataframe.copy()

    cleaned_dataframe["request_text"] = (
        cleaned_dataframe["request_text"]
        .fillna("")
        .str.strip()
        .str.lower()
    )

    cleaned_dataframe["request_text"] = (
        cleaned_dataframe["request_text"].replace("", " ")
    )

    return cleaned_dataframe


def import_and_clean_data():
    """Clean request text by removing excess whitespace and converting to lowercase.

    Missing request descriptions are replaced with a single space."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")

    dataframe = pd.read_csv(RAW_DATA_PATH)

    if not REQUIRED_COLUMNS.issubset(dataframe.columns):
        missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    original_count = len(dataframe)

    cleaned_dataframe = clean_data(dataframe)
    cleaned_dataframe = check_duplicate_data(cleaned_dataframe)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_dataframe.to_csv(CLEAN_DATA_PATH, index=False)

    removed_count = original_count - len(cleaned_dataframe)

    print("Data import and cleaning completed.")
    print(f"Original records: {original_count}")
    print(f"Cleaned records: {len(cleaned_dataframe)}")
    print(f"Duplicate records removed: {removed_count}")
    print(f"Cleaned dataset saved to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    import_and_clean_data()