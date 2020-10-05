import lightgbm as lgb
import numpy as np

class LgbValidator:
    """Wrapper around light GBM for Faron's CrossValidator class"""
    def __init__(self, seed, params):
        self.estimator = lgb
        self.params = params
        self.seed = seed

    def train(self, x, y, x_val, y_val, sample_weights=None):
        np.random.seed(self.seed)

        d_train = lgb.Dataset(x, label=y)
        d_valid = lgb.Dataset(x_val, label=y_val)

        watchlist = d_valid
        self.estimator = self.estimator.train(
            self.params, d_train, 2500, watchlist, early_stopping_rounds=25)


    def predict(self, x):
        preds = self.estimator.predict(x)
        return preds
