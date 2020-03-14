# pipeline_blocks

import numpy as np
from steppy.adapter import Adapter, E
from steppy.base import Step, make_transformer
from attrdict import AttrDict

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from .models import XGBoost, LightGBM, NeuralNetwork, CatBoost, SklearnClassifier
from ..common.utils import get_logger
from sklearn.ensemble import StackingClassifier

logger = get_logger()


def scale_block(suffix):
  name = f'scale{suffix}'

  scale = MinMaxScaler()

  return AttrDict({'name':name, 'transformer': scale})


def lightgbm_block(so_config, suffix, **kwargs):

  logger.info('lightgbm block called!')
  model_name = f'light_gbm{suffix}'

  light_gbm = LightGBM(so_config.light_gbm)
  return AttrDict({'name':model_name, 'transformer':light_gbm})

def catboost_block(so_config, suffix, **kwargs):
  
  model_name = f'catboost{suffix}'
  
  catboost = CatBoost(so_config.catboost)
 
  return AttrDict({'name':model_name, 'transformer':catboost})

def xgboost_block(so_config, suffix, **kawrgs):
  
  model_name = f'xgboost{suffix}'
  
  xgboost = XGBoost(so_config.xgboost)

  return AttrDict({'name':model_name, 'transformer':xgboost})

def neural_network_block(so_config, suffix, **kwargs):
  
  model_name = f'neural_network{suffix}'
  nn = NeuralNetwork(**so_config.neural_network)  
  return AttrDict({'name':model_name, 'transformer':nn})

def sklearn_clf_block(ClassifierClass, clf_name, so_config, suffix, **kawrgs):
  logger.info('sklearn classifier called!')
  model_name = f'{clf_name}{suffix}'
  model_params = getattr(so_config, clf_name)
  sklearn_clf = SklearnClassifier(ClassifierClass, model_params) 
  return AttrDict({'name':model_name, 'transformer':sklearn_clf})
