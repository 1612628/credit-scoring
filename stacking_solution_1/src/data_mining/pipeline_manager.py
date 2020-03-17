# pipeline_manager

import os
import shutil
import json

from attrdict import AttrDict
import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

import gc
gc.enable()

from . import pipeline_config as config

from . import pipeline_blocks as blocks
from .pipelines import PIPELINES 
from ..common.utils import init_logger, read_params, set_seed, param_eval, create_submission, add_prefix_keys
from ..common.custom_plot import CSPlot
    
set_seed(config.RANDOM_SEED)
logger = init_logger()
params = read_params(fallback_file='./credit-scoring/stacking_solution_1/configs/config.yaml')

class PipelineManager:
    def train(self, pipeline_name, data_dev_mode, tag):
        self.clf, self.train_set, self.test_set = train(pipeline_name, data_dev_mode, tag)
    
    def predict(self, pipeline_name, tag, is_submit):
        predict_and_submit(pipeline_name, tag, self.clf, self.test_set, is_submit)

    def tunning(self, pipeline_name, tag):
        hyperparameter_tunning(pipeline_name, False, tag)

def read_data(data_dev_mode, tag):
    data = _read_data(data_dev_mode)
    train_set = data['train'].copy()
    test_set = data['test'].copy()

    logger.info(f'Train shape: {train_set.shape}')
    y = train_set[config.TARGET_COL].values.reshape(-1,)
    train_set = train_set.drop(columns=config.TARGET_COL)

    logger.info('Feature extraction...')
    pca_extract = blocks.pca_block(tag)
    train_new_features = pd.DataFrame(pca_extract.transformer.fit_transform(train_set))
    test_new_features = pd.DataFrame(pca_extract.transformer.fit_transform(test_set))
    train_set = pd.concat([train_set, train_new_features], axis=1)
    test_set = pd.concat([test_set, test_new_features])
    

    logger.info('Feature selection...')
    selection = blocks.selection_block(config.SOLUTION_CONFIG, tag)
    selection.transformer.fit(train_set, y, test_set)
    cols_selected = selection.transformer.transform(train_set)
    logger.info(f'TRAINING, number of features: {len(cols_selected)}')
    
    train_set = train_set[cols_selected]
    train_set[config.TARGET_COL[0]] = y
    test_set = test_set[cols_selected]
    
    del data, y, pca_extract, train_new_features, test_new_features, selection
    gc.collect()

    return train_set, test_set

def train(pipeline_name, data_dev_mode, tag):
    logger.info('TRAINING...')
    
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_dir):
        logger.info('Cleaning experiment directory...')
        shutil.rmtree(params.experiment_dir)

    train_set, test_set = read_data(data_dev_mode, tag) 
    
    y = train_set[config.TARGET_COL].values.reshape(-1,)
    train_set = train_set.drop(columns=config.TARGET_COL)
    
    pipeline = PIPELINES[pipeline_name](so_config = config.SOLUTION_CONFIG, suffix=tag)

    logger.info('Start pipeline fit')
    clf = pipeline.fit(train_set, y)

    logger.info('DONE TRAINING...')

    return clf, train_set, test_set

def predict_and_submit(pipeline_name, suffix, classifier, test_set, is_submit=False):
    logger.info('PREDICT...')
    
    pipeline = classifier 
    logger.info('Start pipeline transform')
    
    y_preds = pipeline.transform(test_set).reshape(-1)
   
    if is_submit:
        logger.info('Creating submission...')
        submission = create_submission(test_set, config.ID_COL[0], config.TARGET_COL[0], y_preds)
        
        submission_filepath = os.path.join(params.experiment_dir,'submission.csv')
        submission.to_csv(submission_filepath, index=None, encoding='utf-8')
        logger.info('Creating submission completed!')
        logger.info(f'submission.csv is pesisted to {submission_filepath}')
    logger.info('DONE PREDICT') 

def hyperparameter_tunning(pipeline_name, data_dev_mode, tag):
    logger.info('HYPERPARAMETER TUNNING...')

    logger.info('Start pipeline')       
    pipeline = PIPELINES[pipeline_name](so_config = config.SOLUTION_CONFIG, suffix=tag)

    logger.info('Create GridSearchCV...')
    param_grid = add_prefix_keys(config.SOLUTION_CONFIG.tuner[pipeline_name], f'{pipeline_name}{tag}__')
    grid = GridSearchCV(estimator=pipeline, 
                        param_grid=param_grid,
                        verbose=1,
                        cv=5,
                        n_jobs=-1)
    train_set, _ = read_data(data_dev_mode, tag) 
    
    y = train_set[config.TARGET_COL].values.reshape(-1,)
    train_set = train_set.drop(columns=config.TARGET_COL)

    logger.info('Start GridSearchCV...')
    grid.fit(train_set, y)

    logger.info('Done GridSearchCV')
    logger.info(f'Best params: {grid.best_params_}')

    json.dumps(grid.best_params_) 

    del train_set, y
    gc.collect()

    logger.info('DONE HYPERPARAMETER TUNNING...')

def _read_data(data_dev_mode):
    logger.info('Reading data...')
    if data_dev_mode:
        nrows = config.DEV_SAMPLE_SIZE
        logger.info(f'Running in "dev-mode" with sample size of {nrows}')
    else:
        nrows = None
    
    raw_data = {}

    logger.info('Reading train ...')
    train = pd.read_csv(params.train_filepath, nrows=nrows)
    raw_data['train']=train
    logger.info('Reading test ...')
    test = pd.read_csv(params.test_filepath, nrows=nrows)
    raw_data['test']=test

    del train, test
    gc.collect()

    logger.info('Reading done!')
    return raw_data

