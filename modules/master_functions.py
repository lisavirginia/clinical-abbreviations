'''
master_functions.py
Library of low-level functions
'''

import pandas as pd
import string

# Function to normalize short form abbreviations
def normalized_short_form(sf):
    # Converts text to uppercase
    sf = sf.upper()
    # Removes leading and trailing whitespace
    sf = sf.strip()
    # Removes punctuation
    sf = sf.translate(str.maketrans('', '', string.punctuation))
    return sf