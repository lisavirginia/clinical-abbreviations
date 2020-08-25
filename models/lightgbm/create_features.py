import re
from typing import Tuple

from fuzzywuzzy import fuzz
import pandas as pd


DATA_PATH = '/ssd-1/clinical/clinical-abbreviations/training/'
OUTPUT_DIR = '/ssd-1/clinical/clinical-abbreviations/data/'
col_1 = 'LF1'
col_2 = 'LF2'

def read_training_data(path: str) -> Tuple:
    """ Reads training data from path.

    Args:
        path: str. Data location
    Returns:
        Tuple of dataframes containing training info
    """
    positives = pd.read_csv(path + 'Train1.csv', sep='|')
    negatives = pd.read_csv(path + 'Train2.csv', sep='|')
    antonyms = pd.read_csv(path + 'Train3.csv', sep='|')
    synonyms = pd.read_csv(path + 'Train4.csv', sep='|')

    train_strings = pd.concat((positives, negatives), axis=0)

    return train_strings, antonyms, synonyms


def create_training_dataframe(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Creates training dataframe filled only with numeric target

    Args:
        raw_dataframe: pd.Dataframe
    Returns:
        target_dataframe: pd.Dataframe
    """

    target_dataframe = raw_dataframe[['Synonym']]
    target_dataframe["target"] = target_dataframe["Synonym"].apply(lambda x: int(x == 'Y'))
    target_dataframe.drop("Synonym", axis=1, inplace=True)

    return target_dataframe


def compare_numeric_values(lf_1, lf_2):
    """ Isolates numeric values in string and checks for equality.

    Args:
        lf_1: str. First term
        lf_2: str. Second term.
    Returns:
        Bool
    """
    numeric_lf_1 = re.sub('[^0-9]', '', lf_1)
    numeric_lf_2 = re.sub('[^0-9]', '', lf_2)

    return int(numeric_lf_1 == numeric_lf_2)


def string_similarity_metrics(
        raw_dataframe: pd.DataFrame, train_dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Compute string similarity metrics and add them to the train dataframe.

    See https://github.com/seatgeek/fuzzywuzzy for more information on the various
    metrics implemented by fuzzywuzzy.

    Args:
        raw_dataframe: df. Strings
        train_dataframe: df. Training data
    Returns:
        train_dataframe: updated training data
    """
    train_dataframe['distance_levenshtein'] = raw_dataframe.apply(
        lambda row: fuzz.ratio(row[col_1], row[col_2]), axis=1)

    train_dataframe['distance_partial_levenshtein'] = raw_dataframe.apply(
        lambda row: fuzz.partial_ratio(row[col_1], row[col_2]), axis=1)

    train_dataframe['distance_token_sort_ratio'] = raw_dataframe.apply(
        lambda row: fuzz.token_sort_ratio(row[col_1], row[col_2]), axis=1)

    train_dataframe['distance_token_set_ratio'] = raw_dataframe.apply(
        lambda row: fuzz.token_set_ratio(row[col_1], row[col_2]), axis=1)

    return train_dataframe


if __name__ == "__main__":

    train_strings, positives, antonyms = read_training_data(DATA_PATH)
    train_dataframe = create_training_dataframe(train_strings)

    train_dataframe['numeric_similarity'] = train_strings.apply(
        lambda row: compare_numeric_values(row[col_1], row[col_2]), axis=1)
    train_dataframe = string_similarity_metrics(train_strings, train_dataframe)

    train_dataframe.to_csv(OUTPUT_DIR + 'full_train.csv')
    train_strings.to_csv(OUTPUT_DIR + 'raw_train.csv')
