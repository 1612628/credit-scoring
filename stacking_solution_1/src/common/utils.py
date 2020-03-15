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
from sklearn.externals import joblib
import sklearn.preprocessing as prep

def create_submission(data, id_name, label_name, label_preds):
  """
  Creating submission pandas dataframe for submitting
  """
  return pd.DataFrame({id_name:[idx+30000 for idx in data.index.tolist()],
                       label_name: label_preds})

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

def read_params(fallback_file):
  """
  Read parameters from ctx or config file
  """
  config = read_yaml(fallback_file)
  params = config.parameters
  
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

