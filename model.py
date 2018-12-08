#xgboost
import pandas as pd
import xgboost
from xgboost import XGBClassifier, plot_importance
from util import plot_roc, plot_importanceFeatures
from sklearn.metrics import accuracy_score

import pickle

class XGBoostModel(object):
    """XGBoost model implementation

    """
    def __init__(self):
        self.clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=1.5, learning_rate=0.02,
       max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,
       n_estimators=600, n_jobs=1, nthread=1, objective='binary:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=0.6)
    
    def train(self, X, y):
        self.clf.fit(X,y)
    
    def predict_proba(self, X):
        y_proba=self.clf.predict_proba(X)
        return y_proba[:, 1]
    
    def predict(self, X):
        y_pred=self.clf.predict(X)
        # print(self.clf.__class__.__name__, accuracy_score(y, y_pred))
        return y_pred

    def feature_impo(self, X):
        print("decreasing importance of features: ")
        feature=pd.DataFrame({'Variable':X.columns,'Importance':self.clf.feature_importances_}).sort_values('Importance', ascending=False)

        # feature = self.clf.feature_importances_
        return feature


    def pickle_clf(self, path = '/Users/shwetasaloni/Documents/git-projects/project577/XGBoost.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))

    def plot_roc(self, X, y, size_x, size_y):
        """Plot the ROC curve for X_test and y_test.
        """
        plot_roc(self.clf, X, y, size_x, size_y)

    def plot_importanceFeatures(self):
        print("plot feature importance:")
        plot_importanceFeatures(self.clf)
