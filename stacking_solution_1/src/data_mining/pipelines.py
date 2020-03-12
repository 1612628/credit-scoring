from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

from . import pipeline_blocks as blocks 

def lightgbm_pipeline(so_config, suffix=''):
   
    light_gbm =  blocks.lightgbm_block(so_config, suffix)

    return light_gbm

def catboost_pipeline(so_config, suffix=''):
   
    catboost =  blocks.catboost_block(so_config, suffix)

    return catboost

def xgboost_pipeline(so_config, suffix=''):
   
    xgboost =  blocks.xgboost_block(so_config, suffix)

    return xgboost

def neural_network_pipeline(so_config, suffix=''):
   
    nn =  blocks.neural_network_block(so_config, suffix)

    return nn

def sklearn_pipeline(so_config, ClassifierClass, clf_name, suffix=''):
   
    sklearn_clf =  blocks.sklearn_clf_block(ClassifierClass, clf_name, so_config, suffix=suffix)

    return sklearn_clf

def stacking_solution_1(so_config, suffix=''):
    """
    There are 2 layers:
      Layer_1: lightgbm, catboost, xgboost, nn, svc, random_forest, logistic, naive_bayes
      Layer_2: LogisticRegression
    """
    classifierclass_tupe_list=[
        (SVC, 'svc'),
        (RandomForestClassifier, 'random_forest'),
        (LogisticRegression, 'log_reg'),
        (BernoulliNB, 'naive_bayes')
    ]
    ensemble_clf = blocks.stacking_solution_1(so_config,suffix,classifierclass_tupe_list)

    return ensemble_clf


PIPELINES = {
    'LightGBM':lightgbm_pipeline,
    'Catboost':catboost_pipeline,
    'XGBoost': xgboost_pipeline,
    'NeuralNetwork':neural_network_pipeline,
    'RandomForest': partial(sklearn_pipeline, ClassifierClass=RandomForestClassifier, clf_name='random_forest'),
    'SVC': partial(sklearn_pipeline, ClassifierClass=SVC, clf_name='svc'),
    'LogisticRegression': partial(sklearn_pipeline, ClassifierClass=LogisticRegression, clf_name='log_reg'),
    'NaiveBayes': partial(sklearn_pipeline, ClassifierClass=BernoulliNB, clf_name='naive_bayes'),
    'StackingSolution1': stacking_solution_1
}