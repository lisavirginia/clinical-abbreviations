# Data Dictionary

Data Field | Name | Description | Example Value
--- | --- | --- | ---
'GroupID' | Group Unique Identifier | Identifies a group of synonymous records | G169326
'RecordID' | Record Unique Identifier | Identifies each record (one per record or row) | R349343
'SF' | Short Form | Abbreviated version of an abbreviation | O.C.
'SFUI' | Short Form Unique Identifier | Identifies a unique short form | S050750
'NormSF' | Normalized Short Form | Lexically normalized version of the short form | oc
'LF' | Long Form | Spelled-out version of an abbreviation | oral contraceptives
'LFUI' | Long Form Unique Identifier | Identifies a unique long form | L121977
'NormLF' | Normalized Long Form | Lexically normalized version of the long form | oral contraceptive
'Source' | Source Inventory | Name of the source sense inventory | ADAM
'Modified' | Modified | Modified by quality control or not | modified

## Auxiliary

These data fields are included in the auxiliary version only. They are unique to a single source (identified in the "Source" column).

Data Field | Name | Description | Source | Example Value
--- | --- | --- | --- | ---
'SFEUI' | Short Form Entry Unique Identifier | Identifies a unique UMLS short form | UMLS-LRABR | E0319213
'LFEUI' | Long Form Entry Unique Identifier | Identifies a unique UMLS long form | UMLS-LRABR | E0044077
'Type' | Type of Entry | Abbreviation or acronym | UMLS-LRABR | acronym
'PrefSF' | Preferred Short Form | Preferred version of a short form | ADAM | o.c.
'Count' | Count  | Number of occurrences in the corpus | ADAM, Vanderbilt | 10
'Score' | Score | Adjusted proportion of occurrences | ADAM | 0.7357
'Frequency' | Frequency | Frequency of the sense in the corpus | Vanderbilt | 0.4168
'UMLS.CUI' | UMLS Concept Unique Identifier | UMLS CUI that mapped to the sense | Vanderbilt | c0009905

*Abbreviations: UMLS, Unified Medical Language System; LRABR, Lexical Resource for Abbreviations and Acronyms; ADAM, Another Database of Abbreviations in Medline*