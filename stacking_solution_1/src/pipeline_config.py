# pipeline_config

from attrdict import AttrDict
from deepsense import neptune

from .utils import read_params, param_eval

ctx = neptune.Context()
params = read_params(ctx, fallback_file='../configs/neptune.yaml')

RANDOM_SEED = 90310
DEV_SAMPLE_SIZE = 500

ID_COL = ['id']
TARGET_COL = ['label']

CATEGORICAL_COLS = ['province', 'district', 'maCv', 'FIELD_7', 'FIELD_8', 'FIELD_9', 'FIELD_10', 'FIELD_13', 'FIELD_35', 
                'FIELD_39', 'FIELD_41', 'FIELD_42', 'FIELD_44']
              
NUMERICAL_COLS = ['age_source1', 'age_source2', 'FIELD_3', 'FIELD_4', 'FIELD_5', 'FIELD_6', 'FIELD_11', 'FIELD_16', 'FIELD_21', 
                  'FIELD_22', 'FIELD_45', 'FIELD_50', 'FIELD_51', 'FIELD_52', 'FIELD_53', 'FIELD_54', 'FIELD_55',  'FIELD_56', 'FIELD_57']

BOOL_COLS = ['FIELD_1', 'FIELD_2', 'FIELD_12', 'FIELD_14', 'FIELD_15', 'FIELD_18', 'FIELD_19',  'FIELD_20', 'FIELD_23', 'FIELD_25', 
             'FIELD_26', 'FIELD_27', 'FIELD_28', 'FIELD_29', 'FIELD_30', 'FIELD_31', 'FIELD_32', 'FIELD_33', 'FIELD_34', 'FIELD_36', 
             'FIELD_37', 'FIELD_38', 'FIELD_46', 'FIELD_47', 'FIELD_48', 'FIELD_49']

SOLUTION_CONFIG = AttrDict({
    'pipeline':{
        'experiment_dir': params.experiment_dir
        },
    'hyperparameter_search':{
        'method': param_eval(params.hyperparameter_search_method),
        'runs': param_eval(params.hyperparameter_search_runs)
        },
    'feature_selection':{
        'use_train': params.use_train
    },
    'preprocessing':{
        'fill_missing': params.fill_missing,
        'fill_value': params.fill_value
    },
    'xgb_preprocessing':{
        'one_hot_encoder':{
            'drop_invariant': True
        }
    },
    'neural_network_preprocessing':{
        'one_hot_encoder':{
            'drop_invariant': True
        },
        'fillna':{
            'fill_value':params.fill_value
        }
    },
    'sklearn_preprocessing':{
        'one_hot_encoder':{
            'drop_invariant': True
        },
        'fillna':{
            'fill_value':params.fill_value
        }
    },
    'train':{
        'table_name':'train',
        'id_col':'id',
        'categorical_cols': CATEGORICAL_COLS,
        'numerical_cols': NUMERICAL_COLS,
    },
    'light_gbm':{
        'device': param_eval(params.lgbm__device),
        'boosting_type': param_eval(params.lgbm__boosting_type),
        'objective': param_eval(params.lgbm__objective),
        'metric': param_eval(params.lgbm__metric),
        'number_boosting_rounds': param_eval(params.lgbm__number_boosting_rounds),
        'early_stopping_rounds': param_eval(params.lgbm__early_stopping_rounds),
        'learning_rate': param_eval(params.lgbm__learning_rate),
        'max_bin': param_eval(params.lgbm__max_bin),
        'max_depth': param_eval(params.lgbm__max_depth),
        'num_leaves': param_eval(params.lgbm__num_leaves),
        'min_child_samples': param_eval(params.lgbm__min_child_samples),
        'subsample': param_eval(params.lgbm__subsample),
        'subsample_freq': param_eval(params.lgbm__subsample_freq),
        'colsample_bytree': param_eval(params.lgbm__colsample_bytree),
        'min_gain_to_split': param_eval(params.lgbm__min_gain_to_split),
        'reg_lambda': param_eval(params.lgbm__reg_lambda),
        'reg_alpha': param_eval(params.lgbm__reg_alpha),
        'is_unbalanced': param_eval(params.lgbm__is_unbalanced),
        'scale_pos_weight': param_eval(params.lgbm__scale_pos_weight),
        'verbose': param_eval(params.lgbm__verbose)
    },
    'catboost': {
        'loss_function': param_eval(params.catboost__loss_function),
        'eval_metric': param_eval(params.catboost__eval_metric),
        'iterations': param_eval(params.catboost__iterations),
        'learning_rate': param_eval(params.catboost__learning_rate),
        'depth': param_eval(params.catboost__depth),
        'l2_leaf_reg': param_eval(params.catboost__l2_leaf_reg),
        'model_size_reg': param_eval(params.catboost__model_size_reg),
        'colsample_bylevel': param_eval(params.catboost__colsample_bylevel),
        'border_count': param_eval(params.catboost__border_count),
        'max_ctr_complexity': param_eval(params.catboost__max_ctr_complexity),
        'od_type': param_eval(params.catboost__od_type),
        'od_wait': param_eval(params.catboost__od_wait),
        'random_seed': RANDOM_SEED,
        'thread_count': params.num_workers,
        'verbose': params.verbose,
    },

    'xgboost': {
        'booster': param_eval(params.xgb__booster),
        'objective': param_eval(params.xgb__objective),
        'tree_method': param_eval(params.xgb__tree_method),
        'eval_metric': param_eval(params.xgb__eval_metric),
        'eta': param_eval(params.xgb__eta),
        'max_depth': param_eval(params.xgb__max_depth),
        'subsample': param_eval(params.xgb__subsample),
        'colsample_bytree': param_eval(params.xgb__colsample_bytree),
        'colsample_bylevel': param_eval(params.xgb__colsample_bylevel),
        'min_child_weight': param_eval(params.xgb__min_child_weight),
        'lambda': param_eval(params.xgb__lambda),
        'alpha': param_eval(params.xgb__alpha),
        'max_bin': param_eval(params.xgb__max_bin),
        'num_leaves': param_eval(params.xgb__max_leaves),
        'nthread': param_eval(params.num_workers),
        'nrounds': param_eval(params.xgb__nrounds),
        'early_stopping_rounds': param_eval(params.xgb__early_stopping_rounds),
        'scale_pos_weight': param_eval(params.xgb__scale_pos_weight),
        'verbose': param_eval(params.verbose),
    },

    'random_forest': {
        'n_estimators': param_eval(params.rf__n_estimators),
        'criterion': param_eval(params.rf__criterion),
        'max_features': param_eval(params.rf__max_features),
        'max_depth': param_eval(params.rf__max_depth),
        'min_samples_split': param_eval(params.rf__min_samples_split),
        'min_samples_leaf': param_eval(params.rf__min_samples_leaf),
        'max_leaf_nodes': param_eval(params.rf__max_leaf_nodes),
        'n_jobs': param_eval(params.num_workers),
        'random_state': RANDOM_SEED,
        'verbose': param_eval(params.verbose),
        'class_weight': param_eval(params.rf__class_weight),
    },

    'log_reg': {
        'penalty': param_eval(params.lr__penalty),
        'tol': param_eval(params.lr__tol),
        'C': param_eval(params.lr__C),
        'fit_intercept': param_eval(params.lr__fit_intercept),
        'class_weight': param_eval(params.lr__class_weight),
        'random_state': RANDOM_SEED,
        'solver': param_eval(params.lr__solver),
        'max_iter': param_eval(params.lr__max_iter),
        'verbose': param_eval(params.verbose),
        'n_jobs': param_eval(params.num_workers),
    },

    'naive_bayes': {
        'alpha': param_eval(params.nb__alpha),
        'binarize': param_eval(params.nb__binarize),
    },

    'neural_network': {
        'architecture_config': {
            'model_params': {
                'layers': param_eval(params.nn__layers),
                'neurons': param_eval(params.nn__neurons),
                'activation': param_eval(params.nn__activation),
                'dropout': param_eval(params.nn__dropout),
                'batch_norm': param_eval(params.nn__batch_norm),
                'l1': param_eval(params.nn__l1),
                'l2': param_eval(params.nn__l2)},
                'optimizer_params': {'lr': param_eval(params.nn__learning_rate),
                'beta_1': param_eval(params.nn__beta_1),
                'beta_2': param_eval(params.nn__beta_2)}
            },
            'training_config': {
                'epochs': param_eval(params.nn__epochs),
                'batch_size': param_eval(params.nn__batch_size)
            },
            'callbacks_config': {},
    },

    'svc': {
        'kernel': param_eval(params.svc__kernel),
        'C': param_eval(params.svc__C),
        'degree': param_eval(params.svc__degree),
        'gamma': param_eval(params.svc__gamma),
        'coef0': param_eval(params.svc__coef0),
        'probability': param_eval(params.svc__probability),
        'tol': param_eval(params.svc__tol),
        'max_iter': param_eval(params.svc__max_iter),
        'verbose': param_eval(params.verbose),
        'random_state': RANDOM_SEED,
    },

    'tuner': {
        'light_gbm': {
            'max_depth': ([3, 8], "choice"),
            'num_leaves': ([20, 100], "choice"),
            'min_child_samples': ([40, 200], "choice"),
            'reg_lambda': ([1e-8, 100.], "log-uniform"),
            'reg_alpha': ([1e-8, 10.], "log-uniform"),
            'min_gain_to_split': ([0., 1], "uniform")
        },
    }
})