import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm

input_loc = "/ssd-1/clinical/clinical-abbreviations/code/Step3Output.csv"
output_dir = "/ssd-1/clinical/clinical-abbreviations/data/full_groups.csv"
df = pd.read_csv(input_loc, sep='|')

def _generate_matches(group):
    match_df = pd.DataFrame(columns=["LF1", "LF2", "RecordID1", "RecordID2"], index=range(len(group)*5))
    current_inx = 0
    for inx, row in group.iterrows():
        group['current_score'] = group.apply(lambda x: fuzz.partial_ratio(x["LF"], row["LF"]), axis=1)
        sorted = group.sort_values(["current_score"], axis=0, ascending=False, inplace=False)

        for sorted_inx, sorted_row in sorted.iterrows():
            if row["RecordID"] == sorted_row["RecordID"]:
                continue
            if sorted_row['current_score'] > 50:
                match_df.loc[current_inx, "LF1"] = row["LF"]
                match_df.loc[current_inx, "LF2"] = sorted_row["LF"]
                match_df.loc[current_inx, "RecordID1"] = row["RecordID"]
                match_df.loc[current_inx, "RecordID2"] = sorted_row["RecordID"]
                current_inx += 1
    match_df = match_df.iloc[:current_inx]
    return match_df, current_inx

groupby = df.groupby(["SF"])
group_pair_dfs = []
for name, group in tqdm(groupby):
    matches, num_matches = _generate_matches(group)
    if num_matches > 0:
        group_pair_dfs.append(matches)

final_df = pd.concat(group_pair_dfs, axis=0, ignore_index=True)
final_df.to_csv(output_dir, index=False)
