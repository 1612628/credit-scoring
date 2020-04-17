# pipeline_blocks

import numpy as np
from attrdict import AttrDict

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV

from .models import XGBoost, LightGBM, NeuralNetwork, CatBoost, SklearnClassifier, FeatureSelection, SMOte, KMeansFeaturizer, Blending, NGBoost
from ..common.utils import get_logger

logger = get_logger()

def pca_block(suffix):
  name = f'pca_{suffix}'

  pca = PCA()

  return AttrDict({'name':name, 'transformer':pca})


def kmeans_block(so_config, suffix):
  name = f'kmeans_{suffix}'

  kmeans = KMeansFeaturizer(**so_config.preprocessing.k_means)

  return AttrDict({'name': name, 'transformer':kmeans})


def over_sample_block(so_config, suffix):
  name = f'smote_{suffix}'
  smote = SMOte(**so_config.preprocessing.smote)

  return AttrDict({'name':name, 'transformer':smote})


def scale_block(suffix):
  name = f'Scale_{suffix}'

  scale = MinMaxScaler()

  return AttrDict({'name':name, 'transformer': scale})

def selection_block(so_config, suffix):
  name = f'FeatureSelection_{suffix}'

  selection = FeatureSelection()

  return AttrDict({'name': name, 'transformer': selection})

def lightgbm_block(so_config, suffix, **kwargs):
  model_name = f'LightGBM_{suffix}'

  light_gbm = LightGBM(**so_config.light_gbm)

  return AttrDict({'name':model_name, 'transformer':light_gbm})

def catboost_block(so_config, suffix, **kwargs):
  
  model_name = f'CatBoost_{suffix}'
  
  catboost = CatBoost(**so_config.catboost)
 
  return AttrDict({'name':model_name, 'transformer':catboost})

def xgboost_block(so_config, suffix, **kawrgs):
  
  model_name = f'XGBoost_{suffix}'
  
  xgboost = XGBoost(**so_config.xgboost)

  return AttrDict({'name':model_name, 'transformer':xgboost})

def neural_network_block(so_config, suffix, **kwargs):
  
  model_name = f'NeuralNetwork_{suffix}'
  nn = NeuralNetwork(**so_config.neural_network)  
  return AttrDict({'name':model_name, 'transformer':nn})

def sklearn_clf_block(ClassifierClass, clf_name, so_config, suffix, **kawrgs):
  logger.info('sklearn classifier called!')
  model_name = f'{clf_name}_{suffix}'
  model_params = getattr(so_config, clf_name)
  sklearn_clf = SklearnClassifier(ClassifierClass, **model_params)
  return AttrDict({'name':model_name, 'transformer':sklearn_clf})

def ngboost_block(so_config, suffix, **kawrgs):
      
  model_name = f'NGBoost_{suffix}'
  
  ngboost = NGBoost(**so_config.ngboost)

  return AttrDict({'name':model_name, 'transformer':ngboost})

def blending(base_models, meta_model, so_config, suffix):
  name = f'blending_{suffix}'
  
  blending = Blending(base_models, meta_model, so_config.blending)
  
  return AttrDict({'name':name, 'transformer': blending})

