import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm

input_loc = "/ssd-1/clinical/clinical-abbreviations/code/Step3Output.csv"
output_dir = "/ssd-1/clinical/clinical-abbreviations/data/full_groups.csv"

PARTIAL_LEVEN_THRESHOLD = 50

def _generate_matches(group):
    """Within a given short form group, generate pairs that are within the desired threshhold"""
    match_df = pd.DataFrame(columns=["LF1", "LF2", "RecordID1", "RecordID2"], index=range(len(group)*5))
    current_inx = 0

    # Iterate through every item in the group
    for inx, row in group.iterrows():
        group['current_score'] = group.apply(lambda x: fuzz.partial_ratio(x["LF"], row["LF"]), axis=1)
        sorted = group.sort_values(["current_score"], axis=0, ascending=False, inplace=False)

        # compare it to all other items in the group
        for sorted_inx, sorted_row in sorted.iterrows():

            # Throw out cases where we are comparing the same item
            if row["RecordID"] == sorted_row["RecordID"]:
                continue

            # Otherwise add the current pair as a match if the partial levenshtien of the 2
            # exceeds teh desired threshold
            if sorted_row['current_score'] > PARTIAL_LEVEN_THRESHOLD:
                match_df.loc[current_inx, "LF1"] = row["LF"]
                match_df.loc[current_inx, "LF2"] = sorted_row["LF"]
                match_df.loc[current_inx, "RecordID1"] = row["RecordID"]
                match_df.loc[current_inx, "RecordID2"] = sorted_row["RecordID"]
                current_inx += 1

    match_df = match_df.iloc[:current_inx]
    return match_df, current_inx

if __name__ == "__main__":

    # Group dataframe by shortform
    df = pd.read_csv(input_loc, sep='|', na_filter=False)
    groupby = df.groupby(["SF"])
    group_pair_dfs = []

    # Iterate through shortform groups and generate matches on them
    for name, group in tqdm(groupby):
        matches, num_matches = _generate_matches(group)
        if num_matches > 0:
            group_pair_dfs.append(matches)

    # Concat and write
    final_df = pd.concat(group_pair_dfs, axis=0, ignore_index=True)
    final_df.to_csv(output_dir, index=False)
