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

class PipelineManager:
    def preprocessing(self, tag, train_filepath=config.params.train_filepath, test_filepath=config.params.test_filepath, train_preprocessed_filepath=config.params.train_preprocessed_filepath, test_preprocessed_filepath=config.params.test_preprocessed_filepath):
        preprocessing(False, tag, train_filepath, test_filepath, train_preprocessed_filepath, test_preprocessed_filepath)

    def train(self, pipeline_name, data_dev_mode, tag, train_filepath=config.params.train_preprocessed_filepath, test_filepath=config.params.test_preprocessed_filepath):
        self.pipe = train(pipeline_name, data_dev_mode, tag, train_filepath, test_filepath)
    
    def predict(self, pipeline_name, tag, is_submit, train_filepath=config.params.train_preprocessed_filepath, test_filepath=config.params.test_preprocessed_filepath):
        predict_and_submit(pipeline_name, tag, self.pipe, train_filepath, test_filepath, is_submit=is_submit)

    def tuning(self, pipeline_name, tag, train_filepath=config.params.train_preprocessed_filepath, test_filepath=config.params.test_preprocessed_filepath):
        hyperparameter_tunning(pipeline_name, False, tag, train_filepath, test_filepath)

def preprocessing(data_dev_mode, tag, train_filepath, test_filepath, train_preprocessed_filepath, test_preprocessed_filepath):
    logger.info('PREPROCESSING...')
    logger.info(f'PREPROCESSING, train filepath: {train_filepath}')
    logger.info(f'PREPROCESSING, test filepath: {test_filepath}')
    
    data = _read_data(data_dev_mode, train_filepath, test_filepath)
    train_set = data['train'].copy()
    test_set = data['test'].copy()

    logger.info(f'PREPROCESSING, Train shape: {train_set.shape}')
    y = train_set[config.TARGET_COL].values.reshape(-1,)
    train_set = train_set.drop(columns=config.TARGET_COL)

    logger.info('PREPROCESSING, Feature extraction...')
    pca_extract = blocks.pca_block(tag)
    train_new_features = pd.DataFrame(pca_extract.transformer.fit_transform(train_set))
    test_new_features = pd.DataFrame(pca_extract.transformer.fit_transform(test_set))
    train_set = pd.concat([train_set, train_new_features], axis=1)
    test_set = pd.concat([test_set, test_new_features], axis=1)

    logger.info('PREPROCESSING, Oversampling...')
    temp_train_set = train_set
    temp_y = y
    over_sampling = blocks.over_sample_block(tag)
    train_set, y = over_sampling.transformer.fit_transform(train_set, y)

    logger.info('PREPROCESSING, Feature selection...')
    selection = blocks.selection_block(config.SOLUTION_CONFIG, tag)
    selection.transformer.fit(train_set, y, test_set)
    cols_selected = selection.transformer.transform(train_set)
    logger.info(f'PREPROCESSING, Feature selection, number of features: {len(cols_selected)}')
    
    train_set = train_set[cols_selected]
    train_set[config.TARGET_COL[0]] = y
    test_set = test_set[cols_selected]
    
    del data, y, pca_extract, train_new_features, test_new_features, selection, temp_train_set, temp_y
    gc.collect()

    logger.info('')
    train_set.to_csv(train_preprocessed_filepath)
    test_set.to_csv(test_preprocessed_filepath)

    logger.info(f'PREPROCESSING, Train set is dumped into path: {train_preprocessed_filepath}')
    logger.info(f'PREPROCESSING, Test set is dumped into path: {test_preprocessed_filepath}')
    logger.info('DONE PREPROCESSING...')
    return train_set, test_set


def train(pipeline_name, data_dev_mode, tag, train_filepath, test_filepath):
    logger.info('TRAINING...')
    
    if bool(config.params.clean_experiment_directory_before_training) and os.path.isdir(config.params.experiment_dir):
        logger.info('Cleaning experiment directory...')
        shutil.rmtree(config.params.experiment_dir)

    data = _read_data(data_dev_mode, train_filepath, test_filepath) 

    train_set = data['train']
    
    y = train_set[config.TARGET_COL].values.reshape(-1,)
    train_set = train_set.drop(columns=config.TARGET_COL)
    
    pipeline = PIPELINES[pipeline_name](so_config = config.SOLUTION_CONFIG, suffix=tag)

    logger.info('TRAINING, Start pipeline fit')
    pipeline.fit(train_set, y)

    logger.info('DONE TRAINING...')
    del data, train_set, y
    gc.collect()
    return pipeline 

def predict_and_submit(pipeline_name, suffix, pipeline, train_filepath, test_filepath, is_submit=False):
    logger.info('PREDICT...')
    
    logger.info('PREDICT, Start pipeline transform')

    data = _read_data(False, train_filepath, test_filepath)

    test_set = data['test']

    y_preds = pipeline.transform(test_set).reshape(-1)
   
    if is_submit:
        logger.info('PREDICT, Creating submission...')
        submission = create_submission(test_set, config.ID_COL[0], config.TARGET_COL[0], y_preds)
        
        submission_filepath = os.path.join(config.params.experiment_dir,'submission.csv')
        submission.to_csv(submission_filepath, index=None, encoding='utf-8')
        logger.info('PREDICT, Creating submission completed!')
        logger.info(f'submission.csv is pesisted to {submission_filepath}')
    logger.info('DONE PREDICT') 

def hyperparameter_tunning(pipeline_name, data_dev_mode, tag, train_filepath, test_filepath):
    logger.info('HYPERPARAMETER TUNNING...')

    logger.info('HYPERPARAMETER TUNNING, Start pipeline')       
    pipeline = PIPELINES[pipeline_name](so_config = config.SOLUTION_CONFIG, suffix=tag)

    logger.info('HYPERPARAMETER TUNNING, Create GridSearchCV...')
    param_grid = add_prefix_keys(config.SOLUTION_CONFIG.tuner[pipeline_name], f'{pipeline_name}{tag}__')
    grid = GridSearchCV(estimator=pipeline, 
                        param_grid=param_grid,
                        verbose=1,
                        cv=5,
                        n_jobs=-1)
    data = _read_data(data_dev_mode, train_filepath, test_filepath) 

    train_set = data['train']

    y = train_set[config.TARGET_COL].values.reshape(-1,)
    train_set = train_set.drop(columns=config.TARGET_COL)

    logger.info('HYPERPARAMETER TUNNING, Start GridSearchCV...')
    grid.fit(train_set, y)

    logger.info('HYPERPARAMETER TUNNING, Done GridSearchCV')
    logger.info(f'HYPERPARAMETER TUNNING, Best params: {grid.best_params_}')

    json.dumps(grid.best_params_) 

    del train_set, y
    gc.collect()

    logger.info('DONE HYPERPARAMETER TUNNING...')

def _read_data(data_dev_mode, train_filepath, test_filepath):
    logger.info('Reading data...')
    if data_dev_mode:
        nrows = config.DEV_SAMPLE_SIZE
        logger.info(f'Running in "dev-mode" with sample size of {nrows}')
    else:
        nrows = None
    
    raw_data = {}

    logger.info('Reading train ...')
    train = pd.read_csv(train_filepath, nrows=nrows)
    raw_data['train']=train
    logger.info('Reading test ...')
    test = pd.read_csv(test_filepath, nrows=nrows)
    raw_data['test']=test

    del train, test
    gc.collect()

    logger.info('Reading done!')
    return raw_data

