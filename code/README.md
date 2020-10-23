# Code for Data Harmonization

**Step 1. Preprocessing** 
Converting source sense inventories to an equivalent format.

**Step 2. Add Data Field**
Addition of lexical normalization and unique identifiers.

**Step 3. Quality Control**
Identification of potential errors and removal or correction of errors.

**Step 4. Remove Redundancy**
Machine learning to identify synonyms and group redundant records.

**Folder: Data**
Contains data used in steps 3 and 4, including:

⋅⋅* clinician-labeled training data
⋅⋅* additional training data, including lists of medical synonyms
⋅⋅* textual replacements of roman numerals, decimals, and common ions
⋅⋅* a medical word corpus for spell-checking
⋅⋅* records of potential and actual errors
