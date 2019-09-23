'''
master_functions.py
Library of low-level functions
'''

import pandas as pd

# Function to clean data frame
def clean(df):
  # Remove leading and trailing white space
  cols = df.select_dtypes(['object']).columns
  df[cols] = df[cols].apply(lambda x: x.str.strip())

# Function to unnest columns in data frame
def expand_col(df, col, d='|'):
  # Split and stack individual entries
  s = df[col].str.split(d).apply(pd.Series, 1).stack()
  # Match up with df indices
  s.index = s.index.droplevel(-1)
  # Name new column
  s.name = col
  # Delete old column
  del df[col]
  # Merge new column with df
  df = df.join(s)
  return df



### TODO ###

# Function to normalize short form abbreviations
def normalized_short_form(sf):
  # Converts text to uppercase
  sf = sf.upper()
  # Removes leading and trailing whitespace
  sf = sf.strip()
  # Removes punctuation
  sf = sf.translate(str.maketrans('', '', string.punctuation))
  return sf

# Function to standardize CUI appearance
def standardize_cui(cui):
  # Use comma delimited CUIs
  cui = cui.replace('|',',')
  # Use CUIs with a capital C
  cui = cui.replace('c', 'C')
  return cui

