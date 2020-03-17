# pipeline_blocks

import numpy as np
from attrdict import AttrDict

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV

from .models import XGBoost, LightGBM, NeuralNetwork, CatBoost, SklearnClassifier, FeatureSelection
from ..common.utils import get_logger

logger = get_logger()

def pca_block(suffix):
  name = f'pca{suffix}'

  pca = PCA()

  return AttrDict({'name':name, 'transformer':pca})


def scale_block(suffix):
  name = f'Scale{suffix}'

  scale = MinMaxScaler()

  return AttrDict({'name':name, 'transformer': scale})

def selection_block(so_config, suffix):
  name = f'FeatureSelection{suffix}'

  selection = FeatureSelection()

  return AttrDict({'name': name, 'transformer': selection})

def lightgbm_block(so_config, suffix, **kwargs):
  model_name = f'LightGBM{suffix}'

  light_gbm = LightGBM(**so_config.light_gbm)

  return AttrDict({'name':model_name, 'transformer':light_gbm})

def catboost_block(so_config, suffix, **kwargs):
  
  model_name = f'CatBoost{suffix}'
  
  catboost = CatBoost(**so_config.catboost)
 
  return AttrDict({'name':model_name, 'transformer':catboost})

def xgboost_block(so_config, suffix, **kawrgs):
  
  model_name = f'XGBoost{suffix}'
  
  xgboost = XGBoost(**so_config.xgboost)

  return AttrDict({'name':model_name, 'transformer':xgboost})

def neural_network_block(so_config, suffix, **kwargs):
  
  model_name = f'NeuralNetwork{suffix}'
  nn = NeuralNetwork(**so_config.neural_network)  
  return AttrDict({'name':model_name, 'transformer':nn})

def sklearn_clf_block(ClassifierClass, clf_name, so_config, suffix, **kawrgs):
  logger.info('sklearn classifier called!')
  model_name = f'{clf_name}{suffix}'
  model_params = getattr(so_config, clf_name)
  sklearn_clf = SklearnClassifier(ClassifierClass, **model_params)
  return AttrDict({'name':model_name, 'transformer':sklearn_clf})
