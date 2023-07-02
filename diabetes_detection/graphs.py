from typing import List, Tuple

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from IPython.display import display


def plot_correlation(df: DataFrame, thresh: float, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Computes the correlation matrix for a pandas DataFrame, and plots a heatmap
    showing only correlations greater than a given threshold (in absolute value).

    Args:
        df (DataFrame): The pandas DataFrame for which to compute correlations.
        thresh (float): The absolute value threshold for correlations to include in the heatmap.
        figsize (Tuple): The size of the output graph. Default is (10, 6).

    Returns:
        None
    """
    # Compute correlations
    corr_matrix = df.corr()

    # Create a mask for values greater than the threshold (in absolute value)
    mask = np.abs(corr_matrix) > thresh

    # Apply the mask to get the new correlation matrix
    high_corr_matrix = corr_matrix[mask]

    # The mask will result in a lot of NaN values, which we'll want to ignore in the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(high_corr_matrix, annot=True, mask=high_corr_matrix.isnull(), cmap='coolwarm')

    # Show the graph
    plt.show()


def calculate_metrics(y_test: np.ndarray, predictions: np.ndarray) -> List[float]:
    """
    Calculates accuracy, precision, recall, and F1-score based on the true labels and
    the predicted labels.

    Args:
        y_test (numpy.ndarray or pd.Series): The true labels.
        predictions (numpy.ndarray or pd.Series): The predicted labels.

    Returns:
        list: A list containing accuracy, precision, recall, and F1-score, in that order.
    """
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return [accuracy, precision, recall, f1]


def display_metrics(y_test: np.ndarray, predictions: np.ndarray) -> None:
    """
    Displays accuracy, precision, recall, and F1-score values in a table format.

    Args:
        y_test (numpy.ndarray or pd.Series): The true labels.
        predictions (numpy.ndarray or pd.Series): The predicted labels.

    Returns:
        None
    """
    metrics_values = calculate_metrics(y_test, predictions)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']

    # Create a DataFrame for displaying the metrics
    df_metrics = pd.DataFrame(metrics_values, index=metrics_names, columns=['Score'])

    # Display the DataFrame
    display(df_metrics)


def plot_confusion_matrix(conf_matrix: np.ndarray, figsize: Tuple[int, int] = (4, 3)) -> None:
    """
    Plots a confusion matrix using seaborn's heatmap.

    Args:
        conf_matrix (numpy.ndarray): The confusion matrix to be plotted.
        figsize (Tuple): The size of the output graph. Default is (4, 3).
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')
    plt.show()
