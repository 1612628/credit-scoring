# pipeline_blocks

import numpy as np
from steppy.adapter import Adapter, E
from steppy.base import Step, make_transformer

from sklearn.linear_model import LogisticRegression
from .models import get_sklearn_classifier, XGBoost, LightGBM, NeuralNetwork,CatBoost

def lightgbm_block(so_config, suffix, **kwargs):
  model_name = f'light_gbm{suffix}'
  
  light_gbm = Step(name=model_name,
                     transformer=LightGBM(name=model_name, **so_config.light_gbm),
                     input_data=['input'],
                     force_fitting=False,
                     experiment_directory=so_config.pipeline.experiment_dir,
                     **kwargs)
  return light_gbm

def catboost_block(so_config, suffix, **kwargs):
  model_name = f'catboost{suffix}'

  catboost = Step(name=model_name,
                  transformer=CatBoost(**so_config.catboost),
                  input_data=['input'],
                  experiment_directory=so_config.pipeline.experiment_dir,
                  **kwargs)
  
  return catboost

def xgboost_block(so_config, suffix, **kawrgs):
  model_name = f'xgboost{suffix}'

  xgboost = Step(name=model_name,
                  transformer=XGBoost(**so_config.xgboost),
                  input_data=['input'],
                  experiment_directory=so_config.pipeline.experiment_dir,
                  **kawrgs)
  
  return xgboost

def neural_network_block(so_config, suffix, **kwargs):
  model_name = f'neural_network{suffix}'

  nn = Step(name=model_name,
            transformer=NeuralNetwork(**so_config.neural_network,
                        suffix=suffix),
            input_data=['input'],
            experiment_directory=so_config.pipeline.experiment_dir,
            **kwargs)
  return nn

def sklearn_clf_block(ClassifierClass, clf_name, so_config, suffix, **kawrgs):
  model_name = f'{clf_name}{suffix}'

  model_params = getattr(so_config, clf_name)

  sklearn_clf = Step(name=model_name,
                     transformer=get_sklearn_classifier(ClassifierClass,**model_params),
                     input_data=['input'],
                     experiment_directory=so_config.pipeline.experiment_dir,
                     **kawrgs)
  
  return sklearn_clf

def stacking_solution_1(so_config, suffix, ClassifierClass_tuple_list):
  
  lightgbm_step = lightgbm_block(so_config, suffix)
  catboost_step = catboost_block(so_config, suffix)
  xgboost_step = xgboost_block(so_config, suffix)
  neural_network_step = neural_network_block(so_config, suffix)
  
  sklearn_clf_steps=[]
  for clf, clf_name in ClassifierClass_tuple_list:
    sklearn_clf_steps.append(sklearn_clf_block(clf, clf_name, so_config, suffix))
  
  gather_step = Step(name='gather_step',
                     transformer=make_transformer(lambda  lst, y: {'input': {'X': np.hstack(lst), 'y': y}}),
                     input_steps=[lightgbm_step, catboost_step, xgboost_step, neural_network_step] + sklearn_clf_steps,
                     input_data=['input'],
                     adapter=Adapter({
                       'lst':[E(lightgbm_step.name,'prediction'), E(catboost_step.name,'prediction'), 
                       E(xgboost_step.name,'prediction'), E(neural_network_step.name,'prediction')]
                       +[E(sklearn_clf.name,'prediction') for sklearn_clf in sklearn_clf_steps],
                       'y':E('input','y')
                     }),
                     experiment_directory=so_config.pipeline.experiment_dir,
                     is_fittable=False)

  ensemble_params = getattr(so_config,'log_reg')
  ensemble_step = Step(name='ensemble_step',
                            transformer = get_sklearn_classifier(LogisticRegression, **ensemble_params),
                            input_data=['input'],
                            input_steps=[gather_step],
                            experiment_directory=so_config.pipeline.experiment_dir,
                            is_fittable=True)

  return ensemble_step

# ++





  





