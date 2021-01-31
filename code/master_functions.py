'''
Master Functions
master_functions.py
'''

import pandas as pd
import string
import subprocess
from configupdater import ConfigUpdater


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


# Function to define normalized short form
def normalized_short_form(sf):
  # Convert to lowercase
  sf = sf.lower()
  # Strip leading and trailing whitespace
  sf = sf.strip()
  # Remove all periods
  sf = sf.replace(".", "")
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


# Function to standardize CUI appearance
def standardize_cui(cui):
  # Use comma delimited CUIs
  cui = cui.replace('|',',')
  # Use CUIs with a capital C
  cui = cui.replace('c', 'C')
  return cui


# Function to add new SFUI
def add_new_SFUI(df_final):
  updater = ConfigUpdater()
  updater.read('setup.cfg')
  # Subset into assigned and unassigned
  df = df_final[df_final['SFUI']=='']
  df_final = df_final[df_final['SFUI']!='']
  if df.empty:
    return df_final
  else:
    # Sort by SF
    df = df.sort_values(by=['SF'])
    df = df.reset_index(drop=True)
    # Assign SFUI
    assignment = int(updater['metadata']['sfui_last_assignment'].value) + 1
    for index, row in df.iterrows():
      if index == 0:
        df['SFUI'].iat[index] = assignment
      elif df['SF'].at[index] == df['SF'].at[index-1]:
        df['SFUI'].iat[index] = assignment
      else:
        assignment += 1
        df['SFUI'].iat[index] = assignment
    # Format SFUI
    df['SFUI'] = 'S' + (df.SFUI.map('{:06}'.format))
    # Add back newly assigned
    df_final = pd.concat([df_final, df])
    df_final = df_final.reset_index(drop=True)
    # Update config file
    updater['metadata']['sfui_last_assignment'].value = assignment
    updater.update_file()
    # Return dataframe
    return df_final


# Function to add new LFUI
def add_new_LFUI(df_final):
  updater = ConfigUpdater()
  updater.read('setup.cfg')
  # Subset into assigned and unassigned
  df = df_final[df_final['LFUI']=='']
  df_final = df_final[df_final['LFUI']!='']
  if df.empty:
    return df_final
  else:
    # Sort by LF
    df = df.sort_values(by=['LF'])
    df = df.reset_index(drop=True)
    # Assign SFUI
    assignment = int(updater['metadata']['lfui_last_assignment'].value) + 1
    for index, row in df.iterrows():
      if index == 0:
          df['LFUI'].iat[index] = assignment
      elif df['LF'].at[index] == df['LF'].at[index-1]:
          df['LFUI'].iat[index] = assignment
      else:
          assignment += 1
          df['LFUI'].iat[index] = assignment
    # Format SFUI
    df['LFUI'] = 'L' + (df.LFUI.map('{:06}'.format))
    # Add back newly assigned
    df_final = pd.concat([df_final, df])
    df_final = df_final.reset_index(drop=True)
    # Update config file
    updater['metadata']['lfui_last_assignment'].value = assignment
    updater.update_file()
    # Return dataframe
    return df_final

