import pandas as pd

TEST_PATH = '/ssd-1/clinical/clinical-abbreviations/data/oof_test.csv'
df = pd.read_csv("/ssd-1/clinical/clinical-abbreviations/data/full_groups.csv")
preds = pd.read_csv(TEST_PATH)

full_df = pd.concat([df, preds], axis=1, ignore_index=True)

full_df.to_csv("/ssd-1/clinical/clinical-abbreviations/data/full_prediction_check.csv", index=False)