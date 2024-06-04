import pandas as pd

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def check_missing_values(df):
    """Check for missing values in the dataframe"""

    # check for missing values
    print(df.isnull().sum())

def remove_duplicates(df):
    """Remove duplicates from the dataframe

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        pd.DataFrame: Dataframe without duplicates
    """
    return df[~df.duplicated(keep='first')]