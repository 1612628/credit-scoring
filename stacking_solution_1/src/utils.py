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
  logger = logging.getLogger('kapala-credit-sccoring')
  logger.setLevel(logging.INFO)
  mess_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s', datefmt='%Y-%m-%d %H-%M-%S')

  # Console handler for validation information
  ch_va = logging.StreamHandler(sys.stdout)
  ch_va.setLevel(logging.INFO)
  ch_va.setFormatter(mess_format)
  logger.addHanlder(ch_va)

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

def calc_rank(preds):
  return (1 + preds.rank().values) / (1+ preds.shape[0])

def chunk_groups(groupby_obj, chunk_size):
  """
  """
  n_groups = groupby_obj.ngroups
  group_chunk, idx_chunk = [], []
  for i, (idx, df) in enumerate(groupby_obj):
    group_chunk.append(df)
    idx_chunk.append(idx)

    if ((i+1) % chunk_size == 0) or (i + 1 == n_groups):
      group_chunk_, idx_chunk_ = group_chunk.copy(), idx_chunk.copy()
      group_chunk, idx_chunk = [], []
      yield group_chunk_, idx_chunk_

def parallel_apply(groups, func, idx_name='Index', num_worker=1, chunk_size=100000):
  """
  """
  n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
  indices, features = [], []
  for idx_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
    with mp.pool.Pool(num_worker) as executor:
      features_chunk = executor.map(func, groups_chunk)
    features.extend(features_chunk)
    indices.extend(idx_chunk)
  
  features = pd.DataFrame(features)
  features.index = indices
  features.index.name = idx_name
  return features

def read_oof_preds(preds_dir, train_filepath, id_col, target_col):
  """
  """
  labels = pd.read_csv(train_filepath, usecols=[id_col, target_col])

  filepaths_train, filepaths_test = [], []

  for filepath in sorted(glob.glob(f'{preds_dir}/*')):
    if filepath.endswith('__oof_train.csv'):
      filepaths_train.append(filepath)
    elif filepath.endswith('__oof_test.csv'):
      filepaths_test.append(filepath)
  
  train_dfs = []
  for filepath in filepaths_train:
    train_dfs.append(pd.read_csv(filepath))
  train_dfs = reduce(lambda df1,df2: pd.merge(df1,df2, on=[id_col, 'fold_id']), train_dfs)
  train_dfs.columns = _clean_columns(train_dfs, keep_colnames=[id_col, 'fold_id'], filepaths=filepaths_train)
  train_dfs = pd.merge(train_dfs, labels, on=[id_col])

  test_dfs = []
  for filepath in filepaths_test:
    test_dfs.append(pd.read_csv(filepath))
  test_dfs = reduce(lambda df1,df2: pd.merge(df1,df2, on=[id_col, 'fold_id']), test_dfs)
  test_dfs.columns = _clean_columns(test_dfs, keep_colnames=[id_col, 'fold_id'], filepaths=filepaths_test)
  test_dfs = pd.merge(test_dfs, labels, on=[id_col])

  return train_dfs, test_dfs

def _clean_columns(df, keep_colnames, filepaths):
  """
  """
  new_colnames = keep_colnames
  features_colnames = df.drop(keep_colnames, axis=1).columns
  for i, colname in enumerate(features_colnames):
    model_name = filepaths[i].split('/')[-1].split('.')[0].replace('__oof_train','').replace('__oof_test','')
    new_colnames.append(model_name)
  return new_colnames

def safe_div(a, b):
  try:
    return float(a) / float(b)
  except:
    return 0.0

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

