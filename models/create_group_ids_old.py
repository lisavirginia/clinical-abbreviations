import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm


MATCH_PATH = "/ssd-1/clinical/clinical-abbreviations/data/full_prediction_check.csv"
RECORD_PATH = "/ssd-1/clinical/clinical-abbreviations/code/Step3Output.csv"
THRESHOLD = .78
eliminate_list = ['ribo', 'non', 'gene', 'acid']

def _remove_suspicious_matches(df_row):
    for item in eliminate_list:
        in_row1 = item in df_row["LF1"]
        in_row2 = item in df_row["LF2"]
        if in_row1 != in_row2:
            return 0
    return df_row['match_score']



full_df = pd.read_csv(MATCH_PATH)
full_df.columns = ['LF1', 'LF2', 'RecordID1', 'RecordID2', 'match_score']
full_df.sort_values(by=['LF1'], axis=0, ascending=True)
full_df = full_df.reset_index(inplace=False, drop=True)

group_ids = pd.concat([full_df['RecordID2'], full_df['RecordID1']]).unique()
group_ids = pd.DataFrame(group_ids, columns=['RecordID'])




match_df = full_df[full_df['match_score'] > .78]
match_df = match_df.reset_index(inplace=False, drop=True)
match_df['match_score'] = match_df.apply(lambda x: _remove_suspicious_matches(x), axis=1)

group_ids['group'] = 0
group_ids.set_index('RecordID', inplace=True)

group_equivalencies = []
cur_group_id = 1
for inx, row in full_df.iterrows():
    if row['match_score'] > THRESHOLD:
        id_1 = row["RecordID1"]
        id_2 = row["RecordID2"]
        if group_ids.loc[id_1, 'group'] == 0 and group_ids.loc[id_2, 'group'] == 0:
            cur_group = cur_group_id
            group_ids.loc[id_1, 'group'] = cur_group
            group_ids.loc[id_2, 'group'] = cur_group
            cur_group_id += 1
        elif group_ids.loc[id_1, 'group'] == 0 and group_ids.loc[id_2, 'group'] != 0:
            cur_group = group_ids.loc[id_2, 'group']
            group_ids.loc[id_1, 'group'] = cur_group

        elif group_ids.loc[id_1, 'group'] != 0 and group_ids.loc[id_2, 'group'] == 0:
            cur_group = group_ids.loc[id_1, 'group']
            group_ids.loc[id_2, 'group'] = cur_group

        else:
            if group_ids.loc[id_1, 'group'] != group_ids.loc[id_2, 'group']:
                group_equivalencies.append([group_ids.loc[id_1, 'group'], group_ids.loc[id_2, 'group']])

group_equivalencies_set = [(min(sample), max(sample)) for sample in group_equivalencies]
group_equivalencies_set = set(group_equivalencies_set)
equivalencies_dict = dict(group_equivalencies_set)

group_ids['group'].replace(equivalencies_dict, inplace=True)
group_ids.reset_index(inplace=True, drop=False)

record_df = pd.read_csv(RECORD_PATH, sep='|')

grouped_df = record_df.merge(group_ids, how='left', on="RecordID")
grouped_df.to_csv("/ssd-1/clinical/clinical-abbreviations/data/Step3Output_with_group_old.csv", index=False)
