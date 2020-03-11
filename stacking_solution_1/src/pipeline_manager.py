# pipeline_manager

import os
import shutil

from attrdict import AttrDict
import numpy as np
import pandas as pd
from scipy.stats import gmean
from deepsense import neptune
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import gc
gc.enable()

from . import pipeline_config as config
from .pipelines import PIPELINES 
from .utils import init_logger, read_params, set_seed, create_submission, verify_submission, calc_rank, read_oof_preds, param_eval

set_seed(config.RANDOM_SEED)
logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx, fallback_file='../configs/neptune.yaml')


class PipelineManager:
    def train(self, pipeline_name, dev_mode, tag):
        train(pipeline_name, dev_mode, tag)
    
    def evaluate(self, pipeline_name, dev_mode, tag):
        evaluate(pipeline_name, dev_mode, tag)
    

def train(pipeline_name, dev_mode, tag):
    logger.info('TRAINING...')
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_dir):
        logger.info('Cleaning experiment directory...')
        shutil.rmtree(params.experiment_dir)
    
    data = _read_data(dev_mode)

    logger.info('Shuffling and spliting into train and dev ...')
    train_set, dev_set = train_test_split(data['train'],
                                                     test_size=params.dev_size,
                                                     random_state=config.RANDOM_SEED,
                                                     shuffle=params.shuffle)
    
    logger.info(f'Train shape: {train_set.shape}')
    logger.info(f'Dev shape: {dev_set.shape}')

    train_data = {
        'input':{
            'X': train_set.drop(columns=[config.TARGET_COL + config.ID_COL]),
            'y': train_set[config.TARGET_COL].values.reshape(-1,1),
            'X_dev': dev_set.drop(columns=[config.TARGET_COL + config.ID_COL]),
            'y_dev': dev_set[config.TARGET_COL].values.reshape(-1,1)
        }
    }

    pipeline = PIPELINES[pipeline_name](config=config.SOLUTION_CONFIG, suffix=tag)

    pipeline.clean_cache()
    logger.info('Start pipeline fit and transform')
    pipeline.fit_transform(train_data)
    pipeline.clean_cache()

def evaluate(pipeline_name, dev_mode, tag):
    logger.info('EVALUATION...')

    data = _read_data(dev_mode)

    logger.info('Shuffling and spliting into train and dev ...')
    _ , dev_set = train_test_split(data['train'],
                                                     test_size=params.dev_size,
                                                     random_state=config.RANDOM_SEED,
                                                     shuffle=params.shuffle)
    logger.info(f'Dev shape: {dev_set.shape}')

    dev_data = {
        'input':{
            'X': dev_set.drop(columns=[config.ID_COL + config.TARGET_COL])
        }
    }

    pipeline = PIPELINES[pipeline_name](config.SOLUTION_CONFIG, suffix=tag)
    pipeline.clean_cache()
    logger.info('Start pipeline transform')
    output = pipeline.transform(dev_data)
    pipeline.clean_cache()

    y_pred = output['prediction']

    logger.info('Calculating AUC on dev set')
    auc_score = roc_auc_score(dev_set[config.TARGET_COL].values.reshape(-1,1), y_pred)
    logger.info(f'ROC AUC score on dev set {auc_score}')
    ctx.channel_send('ROC_AUC',0, auc_score)

def _read_data(dev_mode):
    logger.info('Reading data...')
    if dev_mode:
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