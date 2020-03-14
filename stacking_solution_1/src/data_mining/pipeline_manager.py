# pipeline_manager

import os
import shutil

from attrdict import AttrDict
import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import gc
gc.enable()

from . import pipeline_config as config
from .pipelines import PIPELINES 
from ..common.utils import init_logger, read_params, set_seed, param_eval, create_submission 
from ..common.custom_plot import CSPlot

set_seed(config.RANDOM_SEED)
logger = init_logger()
params = read_params(fallback_file='./credit-scoring/stacking_solution_1/configs/config.yaml')




class PipelineManager:
    def train(self, pipeline_name, data_dev_mode, tag):
        self.clf = train(pipeline_name, data_dev_mode, tag)
    
    def evaluate(self, pipeline_name, data_dev_mode, tag):
        evaluate(pipeline_name, data_dev_mode, tag, self.clf)
    
    def predict(self, pipeline_name, tag, is_submit):
        predict_and_submit(pipeline_name, tag, self.clf, is_submit)
    
def train(pipeline_name, data_dev_mode, tag):
    logger.info('TRAINING...')
    
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_dir):
        logger.info('Cleaning experiment directory...')
        shutil.rmtree(params.experiment_dir)
    
    data = _read_data(data_dev_mode)
    train_set = data['train']
    logger.info(f'Train shape: {train_set.shape}')

    pipeline = PIPELINES[pipeline_name](so_config = config.SOLUTION_CONFIG, suffix=tag)

    logger.info('Start pipeline fit')
    clf = pipeline.fit(train_set.drop(columns=config.TARGET_COL), train_set[config.TARGET_COL].values.reshape(-1))

    logger.info('DONE TRAINING...')

    return clf

def evaluate(pipeline_name, data_dev_mode, tag, classifier):
    logger.info('EVALUATION...')
    
    data = _read_data(data_dev_mode)

    logger.info('Shuffling and spliting into train and dev ...')
    _ , dev_set = train_test_split(data['train'],
                                   test_size=params.dev_size,
                                   random_state=config.RANDOM_SEED,
                                   shuffle=params.shuffle)
    logger.info(f'Dev shape: {dev_set.shape}')

    pipeline = classifier 
    logger.info('Start pipeline transform')
    output = pipeline.transform(dev_set)

    y_pred = output

    logger.info('Calculating AUC on dev set')
    auc_score = roc_auc_score(dev_set[config.TARGET_COL], y_pred)
    logger.info(f'ROC AUC score on dev set {auc_score}')
    logger.info(f'Done EVALUATION')

def predict_and_submit(pipeline_name, suffix, classifier, is_submit=False):
    logger.info('PREDICT...')
    
    data = _read_data(False)

    # train_set = data['train']
    test_set = data['test']

    pipeline = classifier 
    logger.info('Start pipeline transform')
    
    # pipeline.fit(train_set.drop(columns=config.TARGET_COL), train_set[config.TARGET_COL].values.reshape(-1))
    y_preds = pipeline.transform(test_set).reshape(-1)
   
    if is_submit:
        logger.info('Creating submission...')
        submission = create_submission(test_set, config.ID_COL[0], config.TARGET_COL[0], y_preds)
        
        submission_filepath = os.path.join(params.experiment_dir,'submission.csv')
        submission.to_csv(submission_filepath, index=None, encoding='utf-8')
        logger.info('Creating submission completed!')
        logger.info(f'submission.csv is pesisted to {submission_filepath}')
    logger.info('DONE PREDICT') 

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

