from steppy.base import BaseTransformer
import sklearn.preprocessing as prep
import joblib
import pandas as pd
from sklearn.base import BaseEstimator

# Scaler classes

class Normalizer(BaseTransformer):
  def __init__(self, **kwargs):
    super().__init__()
    self.estimator_ = prep.Normalizer() 

  def fit(self, X, **kwargs):
    self.estimator_.fit(X)
    return self
  
  def transform(self, X, **kwargs):
    X = self.estimator_.transform(X)
    return {'X': X}

  def persist(self, filepath):
    joblib.dump(self.estimator_, filepath)

  def load(self, filepath):
    self.estimator_ = joblib.load(filepath)
    return self

# class MinMaxScaler(BaseTransformer):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.estimator_ = prep.MinMaxScaler() 

#     def fit(self, X, **kwargs):
#         self.estimator_.fit(X)
#         return self
  
#     def transform(self, X, **kwargs): 
#         cols = X.columns
#         X_ = self.estimator_.transform(X)
#         return {'X': pd.DataFrame(X_, columns=cols)}

#     def persist(self, filepath):
#         joblib.dump(self.estimator_, filepath)

#     def load(self, filepath):
#         self.estimator_ = joblib.load(filepath)
#         return self


