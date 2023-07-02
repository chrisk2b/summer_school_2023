import pandas as pd
from sklearn.utils import resample


def balance_binary_classes(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """
    Balance dataframe by down-sampling the majority class.

    Args:
        df (pandas.DataFrame): The original dataframe.
        label_column (str): The name of the label column.

    Returns:
        pandas.DataFrame: A new dataframe with balanced classes.

    """

    # Compute majority and minority classes
    class_counts = df[label_column].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    # Separate majority and minority classes
    df_majority = df[df[label_column] == majority_class]
    df_minority = df[df[label_column] == minority_class]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                       replace=False,    # Sample without replacement
                                       n_samples=len(df_minority),     # Match minority class size
                                       random_state=123) # Reproducible results

    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    return df_balanced
