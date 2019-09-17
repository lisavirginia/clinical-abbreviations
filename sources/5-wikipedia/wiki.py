"""
Initial attempt to pull medical abbreviation data from Wikipedia
"""
import requests
import json
import os
import time
import pandas as pd
from string import ascii_uppercase


# Returns a nicely formatted JSON for printing
def dump(js):
    return json.dumps(js, indent = 4)


# Return content not in quotations
def unquoted_content(s):
    while ('(' in s):
        s = s[:s.find('(')].strip() + s[s.find(')')+1:].strip()
    return s.strip()


# Remove bold and italic formatting ticks from wiki markdown language
def remove_wiki_bold_italic(s):
    if "''" in s:
        s = ''.join(s.split("''"))
    elif "'''" in s:
        s = ''.join(s.split("'''"))
    return s


# Remove link formatting from wiki markdown
def remove_wiki_link(string):
    # If it's not a link, carry on
    if '[[' not in string:
        return string
    out_str = ''
    for s in string.split('[['):
        if ']]' in s:
            if '|' in s:
                out_str += s.split('|')[-1]
            else:
                out_str += s
            out_str = out_str.replace(']]', '')
        else:
            out_str += s
    return out_str.strip()



# Placeholder function to "normalize" abbreviations - remove periods, etc.
def normalize_abr(abr):
    if '|' in abr:
        abr = abr.split('|')[-1]
    abr = remove_wiki_bold_italic(abr)
    abr = remove_wiki_link(abr)
    return abr.strip()


# Takes a long form from Wikipedia and removes markdown formatting
def normalize_long_form(long_form):
    # Simple entry without any links
    if '[[' not in long_form:
        return long_form

    # Normalized Long Form return value
    norm_lf = ''

    # Loop through chunks based on when links begin (there may be multiple)
    for s in long_form.split('[['):
        # If this content contains a link
        if ']]' in s:
            # Use the hyperlink label, not they hyperlink
            temp_s = s.split('|')[-1]
            # Remove hyperlink ending brackets
            temp_s = temp_s.replace(']]', '')
            # add content to normalized long form
            norm_lf += temp_s

        # The content before '[[' should be kept
        else:
            norm_lf += s

    # Remove bold and italic formatting from wikipedia
    norm_lf = remove_wiki_bold_italic(norm_lf)

    # Remove content in parentheses
    norm_lf = unquoted_content(norm_lf)

    return norm_lf.strip()


# Given a Wikipedia page title string, return the contents of tables
#   on that page as a list of strings, where each item in the list is
#   a row in the table, stored in wiki markup language
def get_wiki_table(page):
    # Store table contents
    table_content = []

    # AIP request for page
    page_cotent = requests.get('https://en.wikipedia.org/w/api.php',
                                params = {'action':'query',
                                          'prop':'revisions',
                                          'titles':page,
                                          'rvprop':'content',
                                          'format':'json'})

    # print dump(page_cotent.json())
    content = page_cotent.json()['query']['pages']
    content = content[content.keys()[0]]['revisions'][0]["*"]

    # Loop over each line of the page
    for line in content.split('\n'):
        # If the line begins with '| ', then it's a line in a wiki table
        if line[0:2] =='| ':
            table_content.append(line)

    return table_content


# Parses wikipedia table entries that have multiple lines, meaning
#   one abbreviation has more than one long form, or vice versa
def parse_multiple_senses(s):
    if '<br />' in s:
        ret = s.split('<br />')
    elif '<br>' in s:
        ret = s.split('<br>')
    else:
        ret = [s]
    return ret


# Parses a wikipedia medical abbreviation data.
#   Input table is the output format from get_wiki_data(), which is
#   a list, where each entry in the list is a row from the table stored
#   as a string.
# Returns a Pandas Data Frame with 3 colums
#   abr             the abbreviation
#   long_form       the long form, with wiki markdown symbols removed
#   long_form_raw   the long form, in raw wiki markdown format
def parse_wiki_table(wiki_table):
    # wiki_dict = {} # Pandas DF might be better?
    wiki_df = pd.DataFrame(columns = ('abr','long_form','long_form_raw'))
    index = 0

    for row in wiki_table:
        # Parse wikipedia markup - one column for abr, one column for long forms
        abr, long_form = row.strip('|').split('||')
        abr, long_form = abr.strip(), long_form.strip()

        # There may be multiple abrs or long_forms
        abrs = parse_multiple_senses(abr)
        long_forms = parse_multiple_senses(long_form)

        # Add each unique abr/long_form pair to the data frame
        for abr in abrs:
            for long_form in long_forms:

                # Don't include long forms with funky HTML formatting
                if ('<' in long_form) or ('<' in abr):
                    continue

                # Normalize short and long forms
                norm_abr = normalize_abr(abr)
                norm_lf = normalize_long_form(long_form).strip()

                # Don't include blank abrs or long forms
                if (norm_lf == '') or (norm_abr == ''):
                    continue

                # Add content to data frame
                wiki_df.loc[index] = [norm_abr,
                                      norm_lf,
                                      long_form]

                index += 1

                # Also store in dictionary for fun
                # wiki_dict[abr] = long_form

    return wiki_df


# Function to pull and aggregate data from wikipedia.org pages of medical
#   abbreviations. Caches a local text file (valid for 1 hour).
# Returns a list, where each entry in the list is a row from the table stored
#   as a string.
def get_wiki_data():
    # Check if cache file exists and was created less than an hour ago
    if os.path.exists('.wiki_data_cache') and \
        (time.time() - os.path.getmtime('.wiki_data_cache')) < 3600:
        print "Using cached data..."
        cache = open('.wiki_data_cache')
        abr_list = cache.read().strip().split('\n')
        cache.close()
        return abr_list

    else:
        print "Fetching data from Wikipedia.org..."
        abr_list = []

        # Loop over the alphabet
        for letter in ascii_uppercase:
            page = 'List_of_medical_abbreviations:_{}'.format(letter)
            abr_list += get_wiki_table(page)

        # Cache the raw data in a local file
        cache = open('.wiki_data_cache', 'w')
        for abr in abr_list:
            cache.write('{}\n'.format(abr.encode('utf-8')))

        # Reload from cache to resolve utf formatting errors
        cache.close()
        cache = open('.wiki_data_cache')
        abr_list = cache.read().strip().split('\n')
        cache.close()

        return abr_list

if __name__ == '__main__':
    # Pull data from wikipedia
    abr_list = get_wiki_data()
    # Parse data into a data frame
    wiki_df = parse_wiki_table(abr_list)
    # Save the data frame locally!
    out_csv = 'wikipedia_abr_database.csv'
    wiki_df.to_csv(out_csv, index = False)
    print 'Saved to file: {}'.format(out_csv)
