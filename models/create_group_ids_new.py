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


# Read data
full_df = pd.read_csv(MATCH_PATH)
record_df = pd.read_csv(RECORD_PATH, sep='|')

full_df.columns = ['LF1', 'LF2', 'RecordID1', 'RecordID2', 'match_score']
full_df.sort_values(by=['LF1'], axis=0, ascending=True)
full_df = full_df.reset_index(inplace=False, drop=True)


# Match already matched LFEUI entries
lfeui_match_df = record_df.dropna(axis=0, how='any', subset= ['SF', 'LFEUI'], inplace=False)
current_groups = lfeui_match_df[['SF', 'LFEUI']].groupby(['SF', 'LFEUI'], axis=0)['SF'].size().reset_index(name='Size')
current_groups = current_groups[current_groups["Size"] > 1]
cur_group_id = current_groups.shape[0]
current_groups['group'] = range(cur_group_id)
merged_record_df = record_df.merge(current_groups[['SF', 'LFEUI', 'group']], how='left', on=['SF', 'LFEUI'])
merged_record_df['group'].fillna(0, inplace=True)

#Match identical long forms
lf_match_df = merged_record_df[merged_record_df['LFEUI'].isnull()]
current_groups = lf_match_df[['LF', 'group']].groupby(['LF'], axis=0)['LF'].size().reset_index(name='Size')
current_groups = current_groups[current_groups["Size"] > 1]
current_groups['group2'] = range(cur_group_id, cur_group_id + len(current_groups))
cur_group_id = cur_group_id + len(current_groups)

merged_record_df_2 = merged_record_df.merge(current_groups[['LF', 'group2']], how='left', on=['LF'])
merged_record_df_2['group2'].fillna(0, inplace=True)
merged_record_df_2['group'] = merged_record_df_2['group'] + merged_record_df_2['group2']


match_df = full_df[full_df['match_score'] > .78]
match_df = match_df.reset_index(inplace=False, drop=True)
match_df['match_score'] = match_df.apply(lambda x: _remove_suspicious_matches(x), axis=1)

group_ids = merged_record_df_2[['RecordID', 'group']]
group_ids.set_index('RecordID', inplace=True)


group_equivalencies = []
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

grouped_df = record_df.merge(group_ids, how='left', on="RecordID")
grouped_df.to_csv("/ssd-1/clinical/clinical-abbreviations/data/Step3Output_with_group.csv", index=False)
