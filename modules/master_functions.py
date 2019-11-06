'''
master_functions.py
'''

import pandas as pd
import string
import subprocess


# Function to clean data frame
def clean(df):
  # Remove leading and trailing white space
  cols = df.select_dtypes(['object']).columns
  df[cols] = df[cols].apply(lambda x: x.str.strip())


# Function to unnest columns in data frame
def expand_col(df, col, d='|'):
  # Split and stack indivcommitidual entries
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


# Function to standardize CUI appearance
def standardize_cui(cui):
  # Use comma delimited CUIs
  cui = cui.replace('|',',')
  # Use CUIs with a capital C
  cui = cui.replace('c', 'C')
  return cui


# Function to define normalized short form
def normalized_short_form(sf):
  # Convert to lowercase
  sf = sf.lower()
  # Strip leading and trailing whitespace
  sf = sf.strip()
  # Convert all punctuation to underscore
  sf = sf.translate(str.maketrans(string.punctuation, '_'*len(string.punctuation)))
  return sf


# Function to execute command line LVG program
def lvg(input_file, flow, output_file, lvg_path):
    # Specify command
    command = [lvg_path, # Specify path
               '-i:' + input_file, # Input
               '-f:' + flow, # Normalization flow
               '-o:' + output_file, # Output
               '-R:1', # Restrict
               '-n'] # Suppress output
    # Execute command
    lvg_process = subprocess.check_output(command)
    return lvg_process

    