from functools import partial
from joblib import Memory

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import StackingClassifier

from . import pipeline_blocks as blocks
from ..common.utils import get_logger

logger = get_logger()


def lightgbm_pipeline(so_config, suffix=''):
    scale = blocks.scale_block(suffix)
    light_gbm =  blocks.lightgbm_block(so_config, suffix)
    memory = Memory(location=so_config.pipeline.experiment_dir, verbose=10)
    lightgbm_pipe = Pipeline([
        (scale.name, scale.transformer),
        (light_gbm.name, light_gbm.transformer),
    ],
    memory=memory
    )
    return lightgbm_pipe

def catboost_pipeline(so_config, suffix=''):
    scale = blocks.scale_block(suffix)

    catboost =  blocks.catboost_block(so_config, suffix)
    memory = Memory(location=so_config.pipeline.experiment_dir, verbose=10)
    catboost_pipe = Pipeline([
        (scale.name, scale.transformer),
        (catboost.name, catboost.transformer)
    ],
    memory=memory
    )

    return catboost_pipe

def xgboost_pipeline(so_config, suffix=''):
    scale = blocks.scale_block(suffix)

    xgboost =  blocks.xgboost_block(so_config, suffix)
    memory = Memory(location=so_config.pipeline.experiment_dir, verbose=10)
    xgboost_pipe = Pipeline([
        (scale.name, scale.transformer),
        (xgboost.name, xgboost.transformer)
    ],
    memory=memory
    )

    return xgboost_pipe

def neural_network_pipeline(so_config, suffix=''):
    scale = blocks.scale_block(suffix)
    
    nn =  blocks.neural_network_block(so_config, suffix)
    memory = Memory(location=so_config.pipeline.experiment_dir, verbose=10)
    nn_pipe = Pipeline([
        (scale.name, scale.transformer),
        (nn.name, nn.transformer)
    ],
    memory=memory
    )

    return nn_pipe

def sklearn_pipeline(so_config, ClassifierClass, clf_name, suffix=''):
    scale = blocks.scale_block(suffix)
    
    sk_clf =  blocks.sklearn_clf_block(ClassifierClass, clf_name, so_config, suffix)
    memory = Memory(location=so_config.pipeline.experiment_dir, verbose=10)
    sk_clf_pipe = Pipeline([
        (scale.name, scale.transformer),
        (sk_clf.name, sk_clf.transformer)
    ],
    memory=memory
    )

    return sk_clf_pipe 

def stacking_solution_1(so_config, suffix=''):
    """
    There are 2 layers:
      Layer_1: lightgbm, catboost, xgboost, nn, svc, random_forest, logistic, naive_bayes
      Layer_2: LogisticRegression
    """
    scale = blocks.scale_block(suffix)
    light = blocks.lightgbm_block(so_config, suffix)
    xgb = blocks.xgboost_block(so_config, suffix)
    cat = blocks.catboost_block(so_config, suffix)
    nn = blocks.neural_network_block(so_config, suffix)
    rd_fr = blocks.sklearn_clf_block(RandomForestClassifier,'random_forest', so_config, suffix)
    logit = blocks.sklearn_clf_block(LogisticRegression, 'log_reg', so_config, suffix)
    nb = blocks.sklearn_clf_block(BernoulliNB, 'naive_bayes', so_config, suffix)
    
    logger.info('Stacking, create layers...')
    clf_layer_1=[
        (light.name, light.transformer),
        (cat.name, cat.transformer),
        (xgb.name, xgb.transformer),
        (nn.name, nn.transformer),
        (rd_fr.name, rd_fr.transformer),
        (nb.name, nb.transformer)
    ]

    logger.info('Stacking, create stacking classifier')
    stack_clf = StackingClassifier(estimators=clf_layer_1, 
                                    final_estimator= logit.transformer, 
                                    stack_method='predict_proba')
    
    logger.info('Ensemble, creating...')
    memory = Memory(location=so_config.pipeline.experiment_dir, verbose=10)
    ensemble_pipe = Pipeline([
        (scale.name, scale.transformer),
        ('stacking', stack_clf)
    ],
    memory=memory)
    return ensemble_pipe

PIPELINES = {
    'LightGBM':lightgbm_pipeline,
    'Catboost':catboost_pipeline,
    'XGBoost': xgboost_pipeline,
    'NeuralNetwork':neural_network_pipeline,
    'RandomForest': partial(sklearn_pipeline, ClassifierClass=RandomForestClassifier, clf_name='random_forest'),
    'LogisticRegression': partial(sklearn_pipeline, ClassifierClass=LogisticRegression, clf_name='log_reg'),
    'NaiveBayes': partial(sklearn_pipeline, ClassifierClass=BernoulliNB, clf_name='naive_bayes'),
    'StackingSolution1': stacking_solution_1
}