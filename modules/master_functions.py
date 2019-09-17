'''
master_functions.py
Library of commonly used functions for the clinical abbreviation expander
'''

import pandas as pd
import string

# Function to standardize short form abbreviations.
def standard_sf(sf):
    # Converts text to uppercase
    sf = sf.upper()
    # Removes punctuation and whitespace
    sf = sf.translate(None, string.punctuation + " ")
    return sf