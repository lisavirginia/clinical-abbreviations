import re
import string
from typing import Tuple

from fuzzywuzzy import fuzz
from num2words import num2words
import pandas as pd


DATA_PATH = '/ssd-1/clinical/clinical-abbreviations/training/'
OUTPUT_DIR = '/ssd-1/clinical/clinical-abbreviations/data/'
col_1 = 'LF1'
col_2 = 'LF2'
cleaned_col_1 = "LF1_clean"
cleaned_col_2 = "LF2_clean"
punct_chars = ",-()"

def read_training_data(path: str) -> Tuple:
    """ Reads training data from path.

    Args:
        path: str. Data location
    Returns:
        Tuple of dataframes containing training info
    """
    positives = pd.read_csv(path + 'Train1.csv', sep='|')
    negatives = pd.read_csv(path + 'Train2.csv', sep='|')
    additional_1 = pd.read_csv(path + 'Train3.csv', sep='|')
    additional_2 = pd.read_csv(path + 'Train4.csv', sep='|')

    train_strings = pd.concat((positives, negatives), axis=0)

    return train_strings, additional_1, additional_2


def create_training_dataframe(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Creates training dataframe filled only with numeric target

    Args:
        raw_dataframe: pd.Dataframe
    Returns:
        target_dataframe: pd.Dataframe
    """

    target_dataframe = raw_dataframe[['Synonym']]
    target_dataframe["target"] = target_dataframe["Synonym"].apply(
        lambda x: int(x == 'Y')
    )
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


def _replace_characters(raw_dataframe: pd.DataFrame, replace_chars: str, create_cleaned=True):
    """Replace given chars in col1 and col2 and create cleaned_col_1 and 2"""

    if create_cleaned:
        initial_col_1 = col_1
        initial_col_2 = col_2
    else:
        initial_col_1 = cleaned_col_1
        initial_col_2 = cleaned_col_2
    translator = str.maketrans(replace_chars, ' ' * len(replace_chars))
    raw_dataframe[cleaned_col_1] = raw_dataframe[initial_col_1].apply(
        lambda x: x.translate(translator)
    )
    raw_dataframe[cleaned_col_2] = raw_dataframe[initial_col_2].apply(
        lambda x: x.translate(translator)
    )

    return raw_dataframe


def _replace_from_dataframe(raw_dataframe: pd.DataFrame, filename: str, space_pad: bool=False) -> pd.DataFrame:
    """Replace strings from replacement file in col_1 and col_2"""

    _replace_characters(raw_dataframe, punct_chars, create_cleaned=False)

    replacement_df = pd.read_csv(DATA_PATH + filename)
    for inx, row in replacement_df.iterrows():
        replace_1 = str(row['LF1'])
        replace_2 = str(row['LF2'])

        if space_pad:
            replace_1 = " " + replace_1 + " "
            replace_2 = " " + replace_2 + " "
            raw_dataframe[cleaned_col_1] = raw_dataframe[cleaned_col_1].apply(
                lambda x: (" " + x + " ").replace(replace_1, replace_2)
            )
            raw_dataframe[cleaned_col_2] = raw_dataframe[cleaned_col_2].apply(
                lambda x: (" " + x + " ").replace(replace_1, replace_2)
            )

            # Strip the extra added whitespace
            raw_dataframe[cleaned_col_1] = raw_dataframe[cleaned_col_1].apply(
                lambda x: x.strip()
            )
            raw_dataframe[cleaned_col_2] = raw_dataframe[cleaned_col_2].apply(
                lambda x: x.strip()
            )

        else:
            raw_dataframe[cleaned_col_1] = raw_dataframe[cleaned_col_1].apply(
                lambda x: x.replace(replace_1, replace_2)
            )
            raw_dataframe[cleaned_col_2] = raw_dataframe[cleaned_col_2].apply(
                lambda x: x.replace(replace_1, replace_2)
            )

    return raw_dataframe

def _tokenize_and_replace_numbers(text: str):
    """Replaces tokens consisting of only numbers"""
    tokens = text.split(" ")
    for i in range(len(tokens)):
        if tokens[i].isdigit():
            tokens[i] = num2words(int(tokens[i]))

    return " ".join(tokens)


def _replace_numbers(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace strings from replacement file in col_1 and col_2"""

    raw_dataframe[cleaned_col_1] = raw_dataframe[cleaned_col_1].apply(
        lambda x: _tokenize_and_replace_numbers(x)
    )
    raw_dataframe[cleaned_col_2] = raw_dataframe[cleaned_col_2].apply(
        lambda x: _tokenize_and_replace_numbers(x)
    )
    return raw_dataframe


def string_similarity_metrics(
        raw_dataframe: pd.DataFrame, target_dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Compute string similarity metrics and add them to the train dataframe.

    See https://github.com/seatgeek/fuzzywuzzy for more information on the various
    metrics implemented by fuzzywuzzy.

    Args:
        raw_dataframe: df. Raw Strings consisting of columns [col_1, col_2]
        target_dataframe: df. output dataframe containing features
    Returns:
        target_dataframe: updated output dataframe containing features
    """


    target_dataframe['distance_levenshtein'] = raw_dataframe.apply(
        lambda row: fuzz.ratio(row[cleaned_col_1], row[cleaned_col_2]), axis=1
    )

    target_dataframe['distance_partial_levenshtein'] = raw_dataframe.apply(
        lambda row: fuzz.partial_ratio(row[cleaned_col_1], row[cleaned_col_2]), axis=1
    )

    target_dataframe['distance_token_sort_ratio'] = raw_dataframe.apply(
        lambda row: fuzz.token_sort_ratio(row[cleaned_col_1], row[cleaned_col_2]), axis=1
    )

    target_dataframe['distance_token_set_ratio'] = raw_dataframe.apply(
        lambda row: fuzz.token_set_ratio(row[cleaned_col_1], row[cleaned_col_2]), axis=1
    )

    return target_dataframe


if __name__ == "__main__":

    # Create training feats
    train_strings, additional_1, additional_2 = read_training_data(DATA_PATH)
    train_strings = pd.concat([train_strings, additional_2, additional_1], ignore_index=True, axis=0)

    train_strings['LF1'] = train_strings["LF1"].astype(str)
    train_strings['LF2'] = train_strings["LF2"].astype(str)

    train_dataframe = create_training_dataframe(train_strings)

    train_strings = _replace_characters(train_strings, punct_chars)
    train_strings.to_csv(OUTPUT_DIR + 'raw_train_1.csv')
    train_strings = _replace_from_dataframe(train_strings, 'greek_and_molecule_replacements.csv')
    train_strings.to_csv(OUTPUT_DIR + 'raw_train_2.csv')
    train_strings = _replace_from_dataframe(train_strings, 'roman_numeral_replacements.csv', space_pad=True)
    train_strings.to_csv(OUTPUT_DIR + 'raw_train_3.csv')
    train_dataframe['numeric_similarity'] = train_strings.apply(
        lambda row: compare_numeric_values(row[cleaned_col_1], row[cleaned_col_2]), axis=1)

    train_strings = _replace_numbers(train_strings)
    train_dataframe = string_similarity_metrics(train_strings, train_dataframe)

    train_dataframe.to_csv(OUTPUT_DIR + 'full_train.csv')
    train_strings.to_csv(OUTPUT_DIR + 'raw_train.csv')

    '''
    # create testing feats
    test_path = "/ssd-1/clinical/clinical-abbreviations/data/full_groups.csv"
    raw_test_dataframe = pd.read_csv(test_path)

    raw_test_dataframe = _replace_characters(raw_test_dataframe, punct_chars)
    raw_test_dataframe = _replace_from_dataframe(raw_test_dataframe, 'greek_and_molecule_replacements.csv')
    raw_test_dataframe = _replace_from_dataframe(raw_test_dataframe, 'roman_numeral_replacements.csv', space_pad=True)

    raw_test_dataframe['numeric_similarity'] = raw_test_dataframe.apply(
        lambda row: compare_numeric_values(row[col_1], row[col_2]), axis=1)
    test_dataframe = raw_test_dataframe[['numeric_similarity']]

    raw_test_dataframe = _replace_numbers(raw_test_dataframe)
    test_dataframe = string_similarity_metrics(raw_test_dataframe, test_dataframe)

    test_dataframe.to_csv(OUTPUT_DIR + 'full_test.csv')
    '''