# pipeline_manager

import os
import shutil
import json

from scipy import interp
import matplotlib.pyplot as plt

from attrdict import AttrDict
import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.metrics import roc_auc_score, auc, plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

import joblib

import gc
gc.enable()

from . import pipeline_config as config

from . import pipeline_blocks as blocks
from .pipelines import PIPELINES 
from ..common.utils import init_logger, read_params, set_seed, param_eval, create_submission, add_prefix_keys
    
set_seed(config.RANDOM_SEED)
logger = init_logger()

class PipelineManager:
    def preprocessing(self, tag, train_filepath=config.params.train_filepath, test_filepath=config.params.test_filepath, train_preprocessed_filepath=config.params.train_preprocessed_filepath, test_preprocessed_filepath=config.params.test_preprocessed_filepath):
        preprocessing(False, tag, train_filepath, test_filepath, train_preprocessed_filepath, test_preprocessed_filepath)

    def preprocessing_cv(self, data_dev_mode, tag):
        preprocessing(data_dev_mode, tag)

    def train(self, pipeline_name, data_dev_mode, tag, train_filepath=config.params.train_preprocessed_filepath, test_filepath=config.params.test_preprocessed_filepath):
        self.pipe = train(pipeline_name, data_dev_mode, tag, train_filepath, test_filepath)
    
    def train_cv(self, pipeline_name, data_dev_mode, tag):
        train_cv(pipeline_name, data_dev_mode, tag)
    
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
    train_set.to_csv(train_preprocessed_filepath, index=False)
    test_set.to_csv(test_preprocessed_filepath, index=False)

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

def preprocessing_cv(data_dev_mode, tag):
    logger.info('PREPROCESSING CV...')
    
    if bool(config.params.clean_experiment_directory_before_training) and os.path.isdir(config.params.experiment_dir):
        logger.info('Cleaning experiment directory...')
        shutil.rmtree(config.params.experiment_dir)

    kfold = _read_kfold_data(data_dev_mode,
                            config.params.cv_X_train_filepaths,
                            config.params.cv_y_train_filepaths,
                            config.params.cv_X_dev_filepaths,
                            config.params.cv_y_dev_filepaths)
    
    for i in range(0, len(kfold)):
        logger.info(f'PREPROCESSING CV, Fold {i}, Train shape: {kfold[i]["X_train"].shape}')
        logger.info(f'PREPROCESSING CV, Fold {i}, y train shape: {kfold[i]["y_train"].shape}')
        logger.info(f'PREPROCESSING CV, Fold {i}, Dev shape: {kfold[i]["X_dev"].shape}')
        logger.info(f'PREPROCESSING CV, Fold {i}, y dev shape: {kfold[i]["y_dev"].shape}')
        
        logger.info(f'PREPROCESSING CV, Fold {i}, Feature extraction...')
        pca_extract = blocks.pca_block(tag)
        train_new_features = pd.DataFrame(pca_extract.transformer.fit_transform(kfold[i]["X_train"]))
        test_new_features = pd.DataFrame(pca_extract.transformer.fit_transform(kfold[i]["X_dev"]))
        kfold[i]["X_train"]= pd.concat([kfold[i]["X_train"], train_new_features], axis=1)
        kfold[i]["X_dev"]= pd.concat([kfold[i]["X_dev"], test_new_features], axis=1)

        logger.info(f'PREPROCESSING, Fold {i}, Oversampling...')
        temp_train_set = kfold[i]["X_train"]
        temp_y = kfold[i]["y_train"]
        over_sampling = blocks.over_sample_block(tag)
        kfold[i]["X_train"], kfold[i]["y_train"] = over_sampling.transformer.fit_transform(kfold[i]["X_train"], kfold[i]["y_train"])

        logger.info(f'PREPROCESSING, Fold {i}, Feature selection...')
        selection = blocks.selection_block(config.SOLUTION_CONFIG, tag)
        selection.transformer.fit(kfold[i]["X_train"], kfold[i]["y_train"], kfold[i]["X_dev"])
        cols_selected = selection.transformer.transform(kfold[i]["X_train"])
        logger.info(f'PREPROCESSING, Fold {i}, Feature selection, number of features: {len(cols_selected)}')
        kfold[i]["X_train"]= kfold[i]["X_train"][cols_selected]
        kfold[i]["X_dev"]= kfold[i]["X_dev"][cols_selected]
    
        del pca_extract, train_new_features, test_new_features, selection, temp_train_set, temp_y
        gc.collect()

        logger.info('')
        kfold[i]["X_train"].to_csv(config.params.cv_X_train_preprocessed_filepaths[i], index=False)
        kfold[i]["y_train"] = pd.DataFrame(kfold[i]["y_train"])
        kfold[i]["y_train"].to_csv(config.params.cv_y_train_preprocessed_filepaths[i], index=False)
        kfold[i]["X_dev"].to_csv(config.params.cv_X_dev_preprocessed_filepaths[i], index=False)

        logger.info(f'PREPROCESSING CV, Fold {i}, Train set is dumped into path: {config.params.cv_X_train_preprocessed_filepaths[i]}')
        logger.info(f'PREPROCESSING CV, Fold {i}, y train set is dumped into path: {config.params.cv_y_train_preprocessed_filepaths[i]}')
        logger.info(f'PREPROCESSING CV, Fold {i}, Dev set is dumped into path: {config.params.cv_X_dev_preprocessed_filepaths[i]}')
        logger.info(f'DONE PREPROCESSING CV, Fold {i},...')
   
    logger.info('DONE PREPROCESSING CV...')

def train_cv(pipeline_name, data_dev_mode, tag):
    logger.info('TRAINING CV ...')
    
    if bool(config.params.clean_experiment_directory_before_training) and os.path.isdir(config.params.experiment_dir):
        logger.info('Cleaning experiment directory...')
        shutil.rmtree(config.params.experiment_dir)

    pipeline = PIPELINES[pipeline_name](so_config = config.SOLUTION_CONFIG, suffix=tag)

    kfold = _read_kfold_data(data_dev_mode,
                            config.params.cv_X_train_preprocessed_filepaths,
                            config.params.cv_y_train_preprocessed_filepaths,
                            config.params.cv_X_dev_prerprocessed_filepaths,
                            config.params.cv_y_dev_filepaths)

    _cross_validate_auc(pipeline, kfold, features=None)

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

    with open(pipeline_name+'.json', 'a+') as out_params_file:
        json.dump(grid.best_params_, out_params_file) 

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

def _read_kfold_data(data_dev_mode, cv_X_train_filepaths, cv_y_train_filepaths, cv_X_dev_filepaths, cv_y_dev_filepaths):
    logger.info('Reading kfold data...')
    if data_dev_mode:
        nrows = config.DEV_SAMPLE_SIZE
        logger.info(f'Running in "dev-mode" with sample size of {nrows}')
    else:
        nrows = None

    kfold = []

    for i in range(0,len(cv_X_train_filepaths)):
        X_train = pd.read_csv(cv_X_train_filepaths[i], nrows=nrows)
        y_train = pd.read_csv(cv_y_train_filepaths[i], nrows=nrows).values.reshape(-1,)
        X_dev = pd.read_csv(cv_X_dev_filepaths[i], nrows=nrows)
        y_dev = pd.read_csv(cv_y_dev_filepaths[i], nrows=nrows).values.reshape(-1,)
        kfold.append({
            "X_train":X_train,
            "y_train":y_train,
            "X_dev":X_dev,
            "y_dev":y_dev
        })

    logger.info('Done reading kfold data.')
    return kfold

def _cross_validate_auc(model, kfold, features=None, **clf_params):
        train_tprs = []
        train_aucs = []
        train_mean_fpr = np.linspace(0, 1, 100)
        dev_tprs = []
        dev_aucs = []
        dev_mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()
        fig.set_size_inches((16,10))
        for i in range(0, len(kfold)):
          print("{}/{}".format(i+1, len(kfold)))
          kf = kfold[i]
          X_train_kf, y_train_kf = kf["X_train"].copy(), kf["y_train"].copy()
          X_dev_kf, y_dev_kf = kf["X_dev"].copy(), kf["y_dev"].copy()
          if features is not None:
            X_train_kf = X_train_kf[features]
            X_dev_kf = X_dev_kf[features]

          model.fit(X_train_kf, y_train_kf, **clf_params)
          # plot train
          train_display = plot_roc_curve(model, X_train_kf, y_train_kf,
                               name='Train ROC fold {}'.format(i),
                               alpha=0.6, lw=1, ax=ax)
          train_interp_tpr = interp(train_mean_fpr, train_display.fpr, train_display.tpr)
          train_interp_tpr[0] = 0.0
          train_tprs.append(train_interp_tpr)
          train_aucs.append(train_display.roc_auc)
          # plot dev
          dev_display = plot_roc_curve(model, X_dev_kf, y_dev_kf,
                               name='Dev ROC fold {}'.format(i),
                               alpha=0.6, lw=1, ax=ax)
          dev_interp_tpr = interp(dev_mean_fpr, dev_display.fpr, dev_display.tpr)
          dev_interp_tpr[0] = 0.0
          dev_tprs.append(dev_interp_tpr)
          dev_aucs.append(dev_display.roc_auc)
  
        # plot mean train
        train_mean_tpr = np.mean(train_tprs, axis=0)
        train_mean_tpr[-1] = 1.0
        train_mean_auc = auc(train_mean_fpr, train_mean_tpr)
        train_std_auc = np.std(train_aucs)
        ax.plot(train_mean_fpr, train_mean_tpr, color='r',
              label=r'Train Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (train_mean_auc, train_std_auc),
              lw=2, alpha=1)
        # plot mean dev
        dev_mean_tpr = np.mean(dev_tprs, axis=0)
        dev_mean_tpr[-1] = 1.0
        dev_mean_auc = auc(dev_mean_fpr, dev_mean_tpr)
        dev_std_auc = np.std(dev_aucs)
        ax.plot(dev_mean_fpr, dev_mean_tpr, color='b',
              label=r'Dev Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (dev_mean_auc, dev_std_auc),
              lw=2, alpha=1)
  
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
             title="Receiver operating characteristic example")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()
