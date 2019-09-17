'''
master_functions.py
Library of low-level functions
'''

# Function to normalize short form abbreviations
def normalized_short_form(sf):
    # Converts text to uppercase
    sf = sf.upper()
    # Removes leading and trailing whitespace
    sf = sf.strip()
    # Removes punctuation
    sf = sf.translate(maketrans('', '', string.punctuation))
    return sf