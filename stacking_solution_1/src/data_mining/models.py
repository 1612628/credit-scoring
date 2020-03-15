# models
from attrdict import AttrDict
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
import catboost as ctb

from sklearn.base import BaseEstimator, ClassifierMixin

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.regularizers import l1_l2
from keras.optimizers import Adam, SGD

from ..common.utils import get_logger


logger = get_logger()


class SklearnClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, classifier, fit_params):
    logger.info('Inital Sklearn classifier...')
    self.params_ = fit_params
    self.classifier_ = classifier
    self.classes_ = np.array([0,1])

  def fit(self, X, y, *args, **kwargs):
    self.estimator_ = self.classifier_(**self.params_) 
    self.estimator_.fit(X, y)
    return self

  def transform(self, X, *args, **kwargs):
    logger.info(f'Fit.')
    pred = self.estimator_.predict_proba(X)[:, 1].reshape(-1, 1)
    logger.info(f'Done fit.')
    return pred
  
  def predict_proba(self, X, *args, **kwargs):
    return self.transform(X, args, kwargs)
  
  def get_params(self, deep=True):
    return {'classifier':self.classifier_, 'fit_params': self.params_}
  

from copy import deepcopy

class LightGBM(BaseEstimator, ClassifierMixin):

  def __init__(self, params):
    super().__init__()
    logger.info('initializing LightGBM ...')
    self.params_ = params
    self.training_params_ = ['number_boosting_rounds','early_stopping_rounds']
    self.evaluation_function_ = None
    self.classes_ = np.array([0,1])
  
  def get_params(self, deep=True):
    return {'params':self.params_}
  
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
  
  def fit(self, X, y, *args, **kwargs):
    logger.info('LightGBM, fit.')
    evaluation_results = {}
    
    self._check_target_shape_and_type(y, 'y')

    y = self._format_target(y)

    logger.info(f'LightGBM, Training data shape: {X.shape}')
    logger.info(f'LightGBM, Training label shape: {y.shape}')
    
    data_train = lgb.Dataset(data=X, 
                             label=y,
                             )
    self.estimator_ = lgb.train(params=self.model_config, 
                               train_set=data_train, 
                               num_boost_round=self.training_config.number_boosting_rounds,
                               feval=self.evaluation_function_,
                               valid_sets=[data_train],
                               valid_names=['train'],
                               early_stopping_rounds=self.training_config.early_stopping_rounds,
                               evals_result=evaluation_results,
                               verbose_eval=self.model_config.verbose,
                               )
    logger.info('LightGBM, done fit.') 
    return self
  
  def transform(self, X, *args, **kwargs):
    logger.info('LightGBM, transform.')
    logger.info(f'LightGBM, transform, testing shape: {X.shape}')
    pred = self.estimator_.predict(X).reshape(-1)
    logger.info(f'LightGBM, transform, predictions shape: {pred.shape}')
    logger.info('LightGBM, done transform.')
    return pred
  
  def predict_proba(self, X, *args, **kwargs):
    return self.transform(X, args, kwargs)

  def _check_target_shape_and_type(self, target, name):
    if not any([isinstance(target, obj_type) for obj_type in [pd.Series, np.ndarray, list]]):
      raise TypeError(
          f'"target" must be "numpy.ndarray" or "pandas.Series" or "list", got {type(target)} instead.'
      )
    try:
      assert len(target.shape) == 1, f'"{name}" must 1-D. It is {target.shape} instead.'
    except:
      print(f'Cannot determine the shape of {name}.\n\
      Type must be "numpy.ndarray" or "pandas.Series" or "list", got {type(target)} and {target.shape} instead.')
    
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


class XGBoost(BaseEstimator, ClassifierMixin):

  def __init__(self, params):
    logger.info('initializing XGBoost ...')
    self.params_ = params
    self.training_params_ = ['num_boost_round', 'early_stopping_rounds']
    self.evaluation_function_ = None
    self.classes_ = np.array([0,1])
  
  def get_params(self, deep=True):
    return {'params': self.params_}
  
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

  def fit(self, X, y, *args, **kwargs):
    logger.info('XGBoost, fit.')
    logger.info(f'XGBoost, Training data shape: {X.shape}')
    logger.info(f'XGBoost, Training label shape: {y.shape}')

    train = xgb.DMatrix(X, 
                        label=y)

    evaluation_results = {}

    self.estimator_ = xgb.train(params=self.model_config, 
                               dtrain=train, 
                               evals=[(train, 'train')], 
                               evals_result=evaluation_results,
                               num_boost_round=self.training_config.num_boost_round,
                               early_stopping_rounds=self.training_config.early_stopping_rounds, 
                               verbose_eval=self.model_config.verbose,
                               feval=self.evaluation_function_)
    logger.info('XGBoost, done fit.')
    return self
  
  def transform(self, X, *args, **kwargs):
    logger.info('XGBoost, transform.')
    logger.info(f'XGBoost, transform, testing shape: {X.shape}')
    X_DMatrix = xgb.DMatrix(X)
    pred = self.estimator_.predict(X_DMatrix).reshape(-1, 1)
    logger.info(f'XGBoost, transform, predictions shape: {pred.shape}')
    logger.info('XGBoost, done transform.')
    return pred
  
  def predict_proba(self, X, *args, **kwargs):
    return self.transform(X, args, kwargs)


class CatBoost(BaseEstimator, ClassifierMixin):

  def __init__(self, params):
    logger.info('Initializing Catboost...')
    self.params_ = params
    self.estimator_ = ctb.CatBoostClassifier(**params)
    self.classes_ = np.array([0,1])
  
  def get_params(self, deep=True):
    return {'params': self.params_}
  
  def fit(self, X, y, *args, **kwargs):
    logger.info(f'CatBoost, fit') 
    logger.info(f'CatBoost, training data shape {X.shape}')
    logger.info(f'CatBoost, training label shape {y.shape}')

    self.estimator_.fit(X,
                       y,
                       eval_set=[(X, y)])
    logger.info(f'CatBoost, done fit') 
    return self
  
  def transform(self, X, *args, **kwargs):
    logger.info(f'CatBoost, transform') 
    logger.info(f'CatBoost, transform, testing shape: {X.shape}')
    pred = self.estimator_.predict_proba(X)[:,1].reshape(-1, 1)
    logger.info(f'CatBoost, transform, predictions shape: {pred.shape}')
    logger.info(f'CatBoost, done transform') 
    return pred
  
  def predict_proba(self, X, *args, **kwargs):
    return self.transform(X, args, kwargs)

  
class NeuralNetwork(BaseEstimator, ClassifierMixin):
  
  def __init__(self, architecture_config, training_config, callbacks_config):
    logger.info('initializing NeuralNetwork ...')
    self.architecture_config_ = architecture_config
    self.model_params_ = architecture_config['model_params']
    self.optimizer_params_ = architecture_config['optimizer_params']
    self.training_config_ = training_config
    self.callbacks_config_ = callbacks_config
    self.classes_ = np.array([0,1])

  def get_params(self, deep=True):
    return {'architecture_config': self.architecture_config_,
            'training_config': self.training_config_,
            'callbacks_config': self.callbacks_config_
            }
    
  def _build_optimizer(self, **kwargs):
    return Adam(**self.optimizer_params_)
  
  def _build_loss(self, **kwargs):
    return 'binary_crossentropy'
  
  def _build_model(self, input_shape, **kwargs):
    K.clear_session()
    model = Sequential()
    for layer in range(self.model_params_['layers']):
      config = {key: val[layer] for key, val in self.model_params_.items() if key != 'layers'}
      if layer == 0:
        model.add(Dense(config['neurons'],
                        kernel_regularizer=l1_l2(l1=float(config['l1']), l2=float(config['l2'])),
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
    model = self._build_model(input_shape)
    optimizer = self._build_optimizer()
    loss = self._build_loss()
    model.compile(optimizer=optimizer, loss=loss)
    return model
  
  def fit(self, X, y, *args, **kwargs):
    logger.info(f'Neural network, fit') 
    logger.info(f'Neural network, training data shape {X.shape}')
    logger.info(f'Neural network, training label shape {y.shape}')

    self.model = self._compile_model(input_shape=(X.shape[1], ))
    
    self.model.fit(X,
                    y,
                    validation_data=(X, y),
                    verbose=1,
                    **self.training_config_)
    logger.info(f'Neural network, done fit') 
    return self
  
  def transform(self, X, *args, **kwargs):
    logger.info(f'Neural network, transform') 
    logger.info(f'Neural network, transform, testing shape: {X.shape}')
    pred = self.model.predict(X, verbose=1)
    pred = np.array([x[0] for x in pred]).reshape(-1, 1)
    logger.info(f'Neural network, transform, predictions shape: {pred.shape}')
    logger.info(f'Neural network, done transform') 
    return pred
  
  def predict_proba(self, X, *args, **kwargs):
    return self.transform(X, args, kwargs)

