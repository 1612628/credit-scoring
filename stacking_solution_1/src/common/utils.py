# utils

import logging
import os
import random
import sys
import multiprocessing as mp
from functools import reduce

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from attrdict import AttrDict
from steppy.base import BaseTransformer
from sklearn.externals import joblib
import sklearn.preprocessing as prep

def create_submission(meta, preds):
  """
  Creating submission pandas dataframe for submitting
  """
  return pd.DataFrame({'id':meta['id'].tolist(),
                       'label': preds})

def verify_submission(submission, sample_submission):
  """
  Check submission dataframe whether in correct form as sample submission
  """
  assert submission.shape == sample_submission.shape, f'Expected submission to have shape {submission.shape}, but got {sample_submission.shape}'
  for sub_id, sam_sub_id in zip(submission['id'].values, sample_submission['id'].values):
    assert sub_id == sam_sub_id, f'Wrong id: expected {sub_id}, but got {sam_sub_id}'

def get_logger():
  """
  Get current logger
  """
  return logging.getLogger('kapala-credit-scoring')

def init_logger():
  """
  Create a new or get a current logger and set default attributes
  """
  logger = logging.getLogger('kapala-credit-scoring')
  logger.setLevel(logging.INFO)
  mess_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s', datefmt='%Y-%m-%d %H-%M-%S')

  # Console handler for validation information
  ch_va = logging.StreamHandler(sys.stdout)
  ch_va.setLevel(logging.INFO)
  ch_va.setFormatter(mess_format)
  logger.addHandler(ch_va)

  return logger

def read_params(ctx, fallback_file):
  """
  Read parameters from ctx or config file
  """
  if ctx.params.__class__.__name__ == 'OfflineContextParams':
    neptune_config = read_yaml(fallback_file)
    params = neptune_config.parameters
  else:
    params = ctx.params
  
  return params

def read_yaml(filepath):
  """
  Read .yaml file in form of AttrDict for easy use and manipulation
  """
  with open(filepath) as file:
    config = yaml.load(file)
  return AttrDict(config)

def param_eval(param):
  """
  """
  try:
    return eval(param)
  except Exception:
    return param

def persist_evaluation_preds(experiment_dir, y_pred, raw_data, id_col, target_col):
  """
  Persist predictions into raw data
  """
  raw_data.loc[:, 'y_pred'] = y_pred.reshape(-1,1)
  pred_df = raw_data.loc[:, [id_col, target_col, 'y_pred']]
  filepath = os.path.join(experiment_dir, 'evaluation_preds.csv')
  logging.info(f'Evaluation prediction shape {pred_df.shape}')
  pred_df.to_csv(filepath, index = None)

def set_seed(seed=31):
  random.seed(seed)
  np.random.seed(seed)

def flatten_list(l):
  return [item for sublist in l for item in sublist]

# Scaler classes

class Normalizer(BaseTransformer):
  def __init__(self, **kwargs):
    super().__init__()
    self.estimator_ = prep.Normalizer() 

  def fit(self, X, **kwargs):
    self.estimator_.fit(X)
    return self
  
  def transform(self, X, ** kwargs):
    X = self.estimator_.transform(X)
    return {'X': X}

  def persist(self, filepath):
    joblib.dump(self.estimator_, filepath)

  def load(self, filepath):
    self.estimator_ = joblib.load(filepath)
    return self

class MinMaxScaler(BaseTransformer):
  def __init__(self, **kwargs):
    super().__init__()
    self.estimator_ = prep.MinMaxScaler() 

  def fit(self, X, **kwargs):
    self.estimator_.fit(X)
    return self
  
  def transform(self, X, ** kwargs):
    X = self.estimator_.transform(X)
    return {'X': X}

  def persist(self, filepath):
    joblib.dump(self.estimator_, filepath)

  def load(self, filepath):
    self.estimator_ = joblib.load(filepath)
    return self

