# models
from attrdict import AttrDict
from deepsense import neptune
from steppy.base import BaseTransformer
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from sklearn.externals import joblib

from toolkit.sklearn_transformers.models import SklearnClassifier
from toolkit.keras_transformers.models import ClassifierXY

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.regularizers import l1_l2
from keras.optimizers import Adam, SGD

from .utils import get_logger
from .callbacks import neptune_monitor_lgbm, NeptuneMonitor 


logger = get_logger()
ctx = neptune.Context()

def callbacks(channel_prefix):
  neptune_monitor = neptune_monitor_lgbm(channel_prefix)
  return [neptune_monitor]

def get_sklearn_classifier(ClassifierClass, **kwargs):
  class SklearnBinaryClassifier(SklearnClassifier):
    def transform(self, X, y=None, target=1, **kwargs):
      pred = self.estimator.predict_proba(X)[:, target]
      return {'prediction': pred}
  
  return SklearnBinaryClassifier(ClassifierClass(**kwargs))

class XGBoost(BaseTransformer):
  def __init__(self, **params):
    super().__init__()
    logger.info('initializing XGBoost ...')
    self.params_ = params
    self.training_params_ = ['nrounds', ['early_stopping_rounds']]
    self.evaluation_function_ = None
  
  @property
  def model_config(self):
    return AttrDict({
        param: value for param, value in self.params_.items() if param not in self.training_params_
    })
  
  @property
  def training_config(self):
    return AttrDict({
        param: value for param, value in self.params_.items() if param in self.training_params_
    })

  def fit(self, X, y, X_dev, y_dev, **kwargs):
    
    train = xgb.DMatrix(X, 
                        label=y)

    dev = xgb.DMatrix(X_dev, 
                      label=y_dev)

    evaluation_results = {}

    self.estimator_ = xgb.train(params=self.model_config, 
                               dtrain=train, evals=[(train, 'train'), (dev, 'dev')], 
                               evals_result=evaluation_results,
                               num_boost_round=self.training_config.nrounds,
                               early_stopping_rounds=self.training_config.early_stopping_rounds, 
                               verbose_eval=self.model_config.verbose,
                               feval=self.evaluation_function_)
    return self
  
  def transform(self, X):
    X_DMatrix = xgb.DMatrix(X)

    pred = self.estimator_.predict(X_DMatrix)
    return {'prediction':pred}
  
  def load(self, filepath):
    self.estimator_ = xgb.Booster(params=self.model_config)
    self.estimator_.load_model(filepath)
    return self
  
  def persist(self, filepath):
    self.estimator_.save_model(filepath)
  

class LightGBM(BaseTransformer):
  def __init__(self, name=None, **params):
    super().__init__()
    logger.info('initializing LightGBM ...')
    self.params_ = params
    self.training_params_ = ['number_boosting_rounds','early_stopping_rounds']
    self.evaluation_function_ = None
    if params['callback_on']:
      self.callbacks_ = callbacks(channel_prefix=name)
    else:
      self.callbacks_ = []
  
  @property
  def model_config(self):
    return AttrDict({
        param: value for param, value in self.params_.items() if param not in self.training_params_
    })
  
  @property
  def training_config(self):
    return AttrDict({
        param: value for param, value in self.params_.items() if param in self.training_params_
    })
  
  def fit(self, X, y, X_dev, y_dev, **kwargs):
    
    evaluation_results = {}

    self._check_target_shape_and_type(y, 'y')
    self._check_target_shape_and_type(y_dev, 'y_dev')

    y = self._format_target(y)
    y_dev = self._format_target(y)

    logger.info(f'LightGBM, training data shape {X.shape}')
    logger.info(f'LightGBM, dev data shape {X_dev.shape}')
    logger.info(f'LightGBM, training label shape {y.shape}')
    logger.info(f'LightGBM, dev label shape {y_dev.shape}')

    data_train = lgb.Dataset(data=X, 
                             label=y,
                             **kwargs)
    data_dev = lgb.Dataset(data=X_dev, 
                           label=y_dev, 
                           **kwargs)

    self.estimator = lgb.train(params=self.model_config, 
                               train_set=data_train, 
                               num_boost_round=self.training_config.number_boosting_rounds,
                               valid_sets=[data_train, data_dev],
                               valid_names=['train','dev'],
                               feval=self.evaluation_function_,
                               early_stopping_rounds=self.training_config.early_stopping_rounds,
                               evals_result=evaluation_results,
                               verbose_eval=self.model_config.verbose,
                               callbacks=self.callbacks_,
                               **kwargs
                               )
    
    return self
  
  def transform(self, X):
    pred = self.estimator_.predict(X)
    return {'prediction':pred}
  
  def load(self, filepath):
    self.estimator_ = joblib.load(filepath)
    return self
  
  def persist(self, filepath):
    joblib.dump(self.estimator_, filepath)
  
  def _check_target_shape_and_type(self, target, name):
    if not any([isinstance(target, obj_type) for obj_type in [pd.Series, np.ndarray, list]]):
      raise TypeError(
          f'"target" must be "numpy.ndarray" or "pandas.Series" or "list", got {type(target)} instead.'
      )
    try:
      assert len(target.shape) == 1, f'"{name}" must 1-D. It is {len(target.shape)} instead.'
    except:
      print(f'Cannot determine the shape of {name}.\n \
            Type must be "numpy.ndarray" or "pandas.Series" or "list", got {type(target)} instead.')
    
  def _format_target(self, target):
    if isinstance(target, pd.Series):
      return target.values
    elif isinstance(target, np.ndarray):
      return target
    elif isinstance(target, list):
      return np.array(target)
    else:
      raise TypeError(
          f'"target" must be "numpy.ndarray" or "pandas.Series" or "list", got {type(target)} instead.'
      )


class CatBoost(BaseTransformer):
  def __init__(self, **kwargs):
    self.estimator_ = ctb.CatBoostClassifier(**kwargs)
  
  def fit(self, X, y, X_dev, y_dev, **kwargs):
    
    logger.info(f'CatBoost, training data shape {X.shape}')
    logger.info(f'CatBoost, dev data shape {X_dev.shape}')
    logger.info(f'CatBoost, training label shape {y.shape}')
    logger.info(f'CatBoost, dev label shape {y_dev.shape}')

    self.estimator_.fit(X,
                       y,
                       eval_set=[(X, y),(X_dev, y_dev)],
                       )
    return self
  
  def transform(self, X):
    pred = self.estimator_.predict_proba(X)
    return {'prediction':pred}
  
  def load(self, filepath):
    self.estimator_.load_model(filepath)
    return self
  
  def persist(self, filepath):
    self.estimator_.save_model(filepath)


class NeuralNetwork(ClassifierXY):
  def __init__(self, architecture_config, training_config, callback_config, **kwargs):
    super().__init__(architecture_config, training_config, callback_config)
    logger.info('initializing NeuralNetwork ...')
    self.params_ = kwargs
    self.name_= 'NeuralNetwork{}'.format(kwargs['suffix'])
    self.model_params_ = architecture_config['model_params']
    self.optimizer_params_ = architecture_config['optimizer_params']

  def _build_optimizer(self, **kwargs):
    return Adam(**self.optimizer_params_)
  
  def _build_loss(self, **kwargs):
    return 'binary_crossentropy'
  
  def _build_model(self, input_shape, **kwargs):
    K.clear_session()
    model = Sequential()
    for layer in range(self.model_params_['layers']):
      config = {key: val[layer] for key, val in self.model_params_.items() if key != 'layer'}
      if layer == 0:
        model.add(Dense(config['neurons'],
                        kernel_regularizer=l1_l2(l1=config['l1'], l2=config['l2']),
                        input_shape=input_shape
                        ))
      else:
        model.add(Dense(config['neurons'],
                        kernel_regularizer=l1_l2(l1=config['l1'], l2=config['l2'])
                        ))
      if config['batch_norm']:
        model.add(BatchNormalization())
      model.add(Activation(config['activation']))
      model.add(Dropout(config['dropout']))
    
    return model
  
  def _compile_model(self, input_shape):
    model = self._build_model(self.model_params_)
    optimizer = self._build_optimizer()
    loss = self._build_loss()
    model.compile(optimizer=optimizer, loss=loss)
    return model
  
  def _create_callbacks(self, **kwargs):
    neptune = NeptuneMonitor(self.name_)
    return [neptune]
  
  def fit(self, X, y, X_dev, y_dev, *args, **kwargs):
    self.callbacks_ = self._create_callbacks()
    self.model_ = self._compile_model(input_shape=(X.shape[1],))
    
    self.model_.fit(X,
                    y,
                    validation_data=(X_dev, y_dev),
                    verbose=1,
                    callbacks=self.callbacks_,
                    **self.training_config)
    return self
  
  def transform(self, X):
    pred = self.model_.predict(X, verbose=1)
    return {'predicted': np.array([x[0] for x in pred])}
  