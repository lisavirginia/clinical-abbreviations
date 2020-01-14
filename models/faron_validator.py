# -*- coding: utf-8 -*-
"""
@author: Mathias Müller | Faron - kaggle.com/mmueller
"""


import copyreg
import types
from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss


def __pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copyreg.pickle(types.MethodType, __pickle_method)


class CrossValidatorMT(object):
    def __init__(self, clf, clf_params=None, nfolds=5, stratified=True, shuffle=True, seed=0, regression=False, nbags=1,
                 metric=log_loss, average_oof=False, verbose=True):
        self.clf = clf
        self.clf_params = clf_params
        self.nfolds = nfolds
        self.stratified = stratified
        self.seed = seed
        self.regression = regression
        self.nbags = nbags
        self.metric = metric
        self.verbose = verbose
        self.average_oof = average_oof
        self.nclass = None
        self.pdim = None
        self.sample_weights = None
        self.shuffle=shuffle

        self.oof_train = None
        self.oof_test = None
        self.elapsed_time = None
        self.cv_scores = None
        self.cv_mean = None
        self.cv_std = None
        self.folds = None
        self.mean_train = None
        self.mean_test = None

        self.x_train = None
        self.y_train = None
        self.x_test = None

        self.ntrain = None
        self.ntest = None

    def run_cv(self, x_train, y_train, x_test=None, sample_weights=None):
        ts = datetime.now()
        if not isinstance(x_train, np.ndarray):
            x_train = np.array(x_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
        if not isinstance(x_test, np.ndarray) and x_test is not None:
            x_test = np.array(x_test)

        self.sample_weights = sample_weights
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

        if self.verbose:
            if x_test is None:
                print('CrossValidatorMT {0}'.format(x_train.shape))
            else:
                print('CrossValidatorMT {0} {1}'.format(x_train.shape, x_test.shape))

        #pool = Pool(processes=self.nfolds)

        self.ntrain = x_train.shape[0]
        self.ntest = 0 if x_test is None else x_test.shape[0]

        self.nclass = 1 if self.regression else np.unique(y_train).shape[0]
        self.pdim = 1 if self.nclass <= 2 else self.nclass

        if self.stratified:
            folds = StratifiedKFold(n_splits=self.nfolds, shuffle=self.shuffle, random_state=self.seed)
        else:
            folds = KFold(n_splits=self.nfolds, shuffle=self.shuffle, random_state=self.seed)

        oof_train = np.zeros((self.ntrain, self.pdim))
        oof_test = np.zeros((self.ntest, self.pdim))
        oof_test_folds = np.empty((self.nfolds, self.ntest, self.pdim))

        cv_scores = []

        if self.verbose:
            print('{0} Fold CV (seed: {1}, stratified: {2}, nbags: {3}, average oof: {4})' \
                .format(self.nfolds, self.seed, self.stratified, self.nbags, self.average_oof))

        ts_fold = datetime.now()
        #folds_oof = pool.map(self._process_fold, folds)
        folds_oof=[]
        for train_ix, valid_ix in folds.split(range(x_train.shape[0])):
            vals=self._process_fold(train_ix, valid_ix)
            folds_oof.append(vals)
        te_fold = datetime.now()

        for i, (train_index, valid_index) in enumerate(folds.split(range(x_train.shape[0]))):
            y_valid_oof = y_train[valid_index]

            scr = self.metric(y_valid_oof, folds_oof[i][0])
            if self.verbose:
                print('Fold {0:02d}: {1:.12f} ({2})'.format(i + 1, scr, (te_fold - ts_fold)))

            cv_scores.append(scr)
            oof_train[valid_index, :] = folds_oof[i][0]
            oof_test_folds[i, :, :] = folds_oof[i][1]

        if self.ntest > 0:
            if self.average_oof:
                oof_test[:, :] = oof_test_folds.mean(axis=0)
            else:
                oof_bag_test = np.empty((self.nbags, self.ntest, self.pdim))
                for k in range(self.nbags):
                    clf = self.clf(params=self.clf_params.copy(), seed=self.seed + k)
                    clf.train(x_train, y_train, sample_weights=self.sample_weights)
                    oof_bag_test[k, :, :] = clf.predict(x_test).reshape((-1, self.pdim))
                oof_test[:, :] = oof_bag_test.mean(axis=0)

        te = datetime.now()
        elapsed_time = (te - ts)

        self.oof_train = oof_train
        self.oof_test = oof_test
        self.cv_scores = cv_scores
        self.cv_mean = np.mean(cv_scores)
        self.cv_std = np.std(cv_scores)
        self.elapsed_time = elapsed_time
        self.folds = folds
        self.mean_train = np.mean(oof_train, axis=0)
        self.mean_test = np.mean(oof_test, axis=0) if self.ntest > 0 else None

        if self.verbose:
            print('CV-Mean: {0:.12f}'.format(self.cv_mean))
            print('CV-Std:  {0:.12f}'.format(self.cv_std))
            print('Runtime: {0}'.format(elapsed_time))

    def _process_fold(self, train_ix, valid_ix):

        x_train_oof = self.x_train[train_ix]
        y_train_oof = self.y_train[train_ix]
        x_valid_oof = self.x_train[valid_ix]
        y_valid_oof = self.y_train[valid_ix]
        weights = self.sample_weights[train_ix] if self.sample_weights is not None else None

        nvalid_oof = x_valid_oof.shape[0]

        oof_bag_valid = np.empty((self.nbags, nvalid_oof, self.pdim))
        oof_bag_test = np.empty((self.nbags, self.ntest, self.pdim))

        for k in range(self.nbags):
            clf = self.clf(params=self.clf_params.copy(), seed=self.seed + k)
            clf.train(x_train_oof, y_train_oof, x_valid_oof, y_valid_oof, sample_weights=weights)
            oof_bag_valid[k, :, :] = clf.predict(x_valid_oof).reshape((-1, self.pdim))
            if self.ntest > 0:
                oof_bag_test[k, :, :] = clf.predict(self.x_test).reshape((-1, self.pdim))

        pred_oof_valid = oof_bag_valid.mean(axis=0)
        pred_oof_test = oof_bag_test.mean(axis=0) if self.ntest > 0 else None

        return pred_oof_valid, pred_oof_test

    @property
    def oof_predictions(self):
        return self.oof_train, self.oof_test

    @property
    def cv_stats(self):
        return self.cv_mean, self.cv_std

    @property
    def oof_means(self):
        return self.mean_train, self.mean_test