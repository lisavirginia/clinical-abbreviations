# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:01:36 2017

@author: Raymond
"""

import pandas as pd
import sklearn.metrics as mt

from faron_validator import CrossValidatorMT
from model_helpers import LgbValidator
import parameter_dicts as params


CURRENT_PARAMS = params.parameters_v1
TRAIN_PATH = '/ssd-1/clinical/clinical-abbreviations/data/full_train.csv'


def load_data(filename):
    """Load train from file and parse out target"""

    train_dataframe = pd.read_csv(filename)
    target = train_dataframe['target']
    train_dataframe.drop('target', axis=1, inplace=True)

    return train_dataframe, target


def run_lgb_models(train_df, target):
    """Run K-folded light GBM model"""

    clf = CrossValidatorMT(
        clf=LgbValidator,
        clf_params=CURRENT_PARAMS,
        nfolds=5,
        stratified=False,
        shuffle=True,
        seed=117,
        regression=False,
        nbags=1,
        metric=mt.log_loss,
        average_oof=True,
        verbose=True
    )

    clf.run_cv(train_df, target)
    return clf


if __name__ == "__main__":

    train_df, target = load_data(TRAIN_PATH)
    clf = run_lgb_models(train_df, target)

    print('F1: ', mt.f1_score(target, clf.oof_predictions[0] > .5))
    RAW_PATH = '/ssd-1/clinical/clinical-abbreviations/data/raw_train.csv'
    raw_data = pd.read_csv(RAW_PATH)
    raw_data['target'] = target
    raw_data['predictions'] = clf.oof_predictions[0].reshape(-1,)
    raw_data.to_csv('/ssd-1/clinical/clinical-abbreviations/data/prediction_check.csv')
