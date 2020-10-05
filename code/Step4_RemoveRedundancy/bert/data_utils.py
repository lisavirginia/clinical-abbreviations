import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MAX_SEQ_LENGTH = 20


    
def _tokenize_string(tokenizer, text):
    """Tokenizes a string using the provided tokenizer"""
    
    cls_token = tokenizer.encode(tokenizer.cls_token)[0]
    eos_token = tokenizer.encode(tokenizer.eos_token)[0]

    if len(text.split()) > MAX_SEQ_LENGTH:
        raise AssertionError("Passed text that contains too many tokens to tokenizer. \
                             Max tokens: {}. Passed tokens: {}".format(MAX_SEQ_LENGTH, len(text.split())))
    tokenized_sequence = [tokenizer.encode(word) for word in text.split()]
    flattened_tokenized_sequence = []
    for word in tokenized_sequence:
        for token_num in word:
            flattened_tokenized_sequence.append(token_num)

    encoded_text = [cls_token] + flattened_tokenized_sequence + [eos_token]
    return encoded_text    
        
def _tokenize_train_data(df, tokenizer):
    """Loads the conll data into lists recursively"""

    df["tokenized_1"] = df["LF1"].apply(lambda x: _tokenize_string(tokenizer, x))
    df["tokenized_2"] = df["LF2"].apply(lambda x: _tokenize_string(tokenizer, x))

    return df

def _create_labels(df):
    """Transforms labels from Y/N to int"""
    
    df["label"] = df["Synonym"].apply(lambda x: int(x == "Y"))
    return df

def load_data(data_path, tokenizer):
    """Loads train data and tokenizes with Roberta"""
    # Initialize lists to hold our data in
    
    df = pd.read_csv(data_path)

    expected_columns = ["LF1", "LF2", "Synonym"]
    if len(df.columns) !=3:
        raise AssertionError("Passed dataframe with incorrect number of columns. Expected 3."
                             "Recieved columns: {}".format(df.columns))
    if sum(df.columns == expected_columns) != 3:
        raise AssertionError("Loaded dataframe does not match training data format. Expected columns: {} \
                             , received columns: {}.".format(expected_columns, df.columns))
      
    tokenized_df = _tokenize_train_data(df, tokenizer)
    
    df_with_labels = _create_labels(tokenized_df)
    
    
    tokenized_array_1 = np.zeros((len(df), MAX_SEQ_LENGTH))
    tokenized_array_2 = np.zeros((len(df), MAX_SEQ_LENGTH))
    tokenized_label_array = np.zeros((len(df), 1))
    
    for inx, (tokens_1, tokens_2, labels) in enumerate(zip(df["tokenized_1"], df["tokenized_2"], df["label"])):
        tokenized_array_1[inx, -min(len(tokens_1), MAX_SEQ_LENGTH):] = tokens_1[:MAX_SEQ_LENGTH]
        tokenized_array_2[inx, -min(len(tokens_2), MAX_SEQ_LENGTH):] = tokens_2[:MAX_SEQ_LENGTH]
        tokenized_label_array[inx, 0] = labels
        
    return df_with_labels[["tokenized_1", "tokenized_2", "label"]], tokenized_array_1,\
        tokenized_array_2, tokenized_label_array
    

class MatchingDataset(Dataset):
    """NER dataset."""

    def __init__(self, data_path, feature_array, tokenizer):
        """
        Args:
            data_path (string): Path to the train csv
            tokenizer: Model-specific tokenizer (from huggingface)
        """
        self.tokenizer = tokenizer   
        self.train_df, self.tokenized_text_1, self.tokenized_text_2, self.labels = load_data(data_path, tokenizer)
        self.feature_array = feature_array

        if len(self.feature_array) != len(self.labels):
            raise AssertionError("Passed incorrect number of additional features"
                                 "Received {}, expected {}.".format(len(self.feature_array), len(self.labels)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx_list):
        if torch.is_tensor(idx_list):
            idx_list = idx_list.tolist()

        sample_text_1 = torch.LongTensor(self.tokenized_text_1[idx_list])
        sample_text_2 = torch.LongTensor(self.tokenized_text_2[idx_list])
        additional_feats = torch.FloatTensor(self.feature_array[idx_list])
        sample_labels = torch.FloatTensor(self.labels[idx_list])
        sample = {'text_1': sample_text_1, 'text_2': sample_text_2, 'labels': sample_labels,
                  'additional_feats': additional_feats}

        return sample