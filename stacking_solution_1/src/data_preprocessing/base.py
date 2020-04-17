import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from ..data_mining.models import NGBoost
import matplotlib.pyplot as plt
import math
import gc
gc.enable()

def _isPattern(value,patterns):
    """_isPattern
    Checking whether value is one of given patterns 
    Parameters
    ----------
    value: object(int, float, str), required
      Value need to check
    patterns: list (['None','na']), required
      patterns
    Return
    ----------
    True: if value in patterns
    False: vice versa
    """
    return (value in patterns)

def _isMissing(value,missing_patterns):
    """_isMissing
    Checking whether value is missing value
    Parameters
    ----------
    value: object(int, float, str), required
      Value need to check
    missing_patterns: list (['None','na']), required
      Missing patterns act as missing values
    Return
    ----------
    True: if value in missing_patterns or isnan
    False: vice versa
    """
    return _isPattern(value, missing_patterns) or (type(value) == float and np.isnan(value))

def _missValueTable(df, missing_patterns, verbosity = False):
    """MissValueTable
    Calculate missing values (np.nan and missing_patterns ['None','ee']) of features in dataframe
    Parameters
    ----------
    df: DataFrame, required
      The dataframe used to calculate missing values and missing rate
    missing_patterns: list (['None','na']), required
      List missing possible patterns act as missing values (null, None, nan, na, NaN,...etc)
    Return
    ----------
    miss_values_table: DataFrame ('Total missing values','Missing values percentage')
      The dataframe with missing values statistic
    """
    # Validate variable information
    assert (type(df) == pd.DataFrame), "df must be pandas DataFrame"
    assert (type(missing_patterns) == list), "missing_patterns must be list"
    # Total missing values
    total_miss_values = pd.Series([])
    # Iterate through columns name
    for col_name in df:
        sr = pd.Series({col_name:df[df.apply(lambda row: _isMissing(row[col_name],missing_patterns), axis=1)].shape[0]})
        total_miss_values = total_miss_values.append(sr)
  
    # Drop feauture without having missing values
    total_miss_values.where(cond = total_miss_values>0,inplace = True)
    total_miss_values.dropna(inplace=True)
    # Percentage of missing value
    percent_miss_values = total_miss_values / df.shape[0]
    # Make a table with resutls
    miss_values_table = pd.concat([total_miss_values, percent_miss_values],axis=1)
    # Change table column names
    miss_values_table.columns = ['Total missing values','Missing values percentage']
    # Sort the table by descending of missing values percentage
    miss_values_table = miss_values_table.sort_values(by=['Missing values percentage'],ascending=False)
    # Print some sumary information
    if verbosity == True:
        print(f'Your selected dataframe has {df.shape[1]} columns.\nThere are {miss_values_table.shape[0]} columns that have missing values')
    return miss_values_table

def DropFeaturesByMissingRate(train_data, test_data,missing_rate):
    """DropFeatureWithMissingRate
    This function used to drop features that have missing rate equal and over missing_rate
    Parameters
    ----------
    train_data: DataFrame (required)
      The train dataframe need to drop features
    test_data: DataFrame (required)
      The test dataframe need to drop features
    missing_rate: float (required)
      The rate of missing values in each features
    Return
    ----------
    data: DataFrame
      The dropped-features dataframe
    """
    # Validate variables information
    assert(type(train_data) == pd.DataFrame),"train_data must be pandas DataFrame"
    assert(type(test_data) == pd.DataFrame),"test_data must be pandas DataFrame"
    assert(type(missing_rate) == float),"missing_rate must be float"
    miss_values = _missValueTable(train_data,['None','na','Nan','NaN'])
    drop_features = miss_values[miss_values[miss_values.columns[1]] >= missing_rate].index.values.tolist()
    train_data_cp = train_data.drop(columns = drop_features).copy()
    test_data_cp = test_data.drop(columns = drop_features).copy()
    return train_data_cp, test_data_cp


def high_corr(data,threshold=0.8):
    corr = data.corr()

    high_corr_dic = {col:[] for col in list(corr.columns)}
    for i in corr.index:
        for j in corr.loc[i,:].index:
            if (i != j) and (corr.loc[i,j] >= 0.8) and (len(high_corr_dic[j]) == 0):
                high_corr_dic[i].append(j)
    return high_corr_dic


def drop_high_corr(train_data, test_data, threshold=0.8):

    train_drop = train_data.copy()
    to_drop = []
    while True:
        high_corr_dic = high_corr(train_drop, threshold)
        high_corr_dic = sorted(high_corr_dic.items(), key=lambda x:len(x[1]), reverse=True)
        if (len(high_corr_dic) <= 0) or (len(high_corr_dic[0][1]) <= 0):
            break
        to_drop.append(high_corr_dic[0][0])
        train_drop = train_drop.drop(columns=[high_corr_dic[0][0]])
    test_drop = test_data.drop(columns=to_drop)
    return train_drop, test_drop

def GenerateIsMissFeatures(data):
    """GenerateIsMissFeatures
    This function aims to generate is_missing features for each features in the given data

    Parameters
    ----------
    data: DataFrame/Series (required)
      The data set that need to generate is_missing features

    Return
    ----------
    data_gen: DataFrame
      The is_missing features DataFrame/Series
    """

    # Validate variables
    assert(data is not None),'data must be specified'

    # Create ismiss data
    bad = data.isnull()
    mapper = {False:0,True:1}
    for col_na in list(bad.columns):
        bad[col_na] = bad[col_na].map(mapper)

    cols_na=[]
    # Rename columns
    if type(data) == pd.DataFrame:
        for col_na in list(data.columns):
            cols_na.append(str(col_na)+'_ismiss')
        bad.columns = cols_na
    elif type(data) == pd.Series:
        bad.rename_axis(str(bad.name)+'_ismiss')

    return bad


def replaceAll(text, rep_dict):
    for key,value in rep_dict.items():
        text = text.replace(key,value)
    return text


def pca(train, test):
    pca = PCA(n_components=0.97, svd_solver='full')
    train_pca = pd.DataFrame(pca.fit_transform(train), columns = ['pca_' + str(i) for i in range(pca.n_components_)])
    test_pca = pd.DataFrame(pca.transform(test), columns = ['pca_' + str(i) for i in range(pca.n_components_)])
    return (train_pca, test_pca)

def one_hot_encoding(X_train, X_test):
    train = X_train.copy()
    test = X_test.copy()
    enc = OneHotEncoder(handle_unknown='ignore')

    train_ohe = pd.DataFrame()
    test_ohe = pd.DataFrame()

    for fea in train:
        train[fea] = train[fea].replace(to_replace=[np.nan], value='none')
        test[fea] = test[fea].replace(to_replace=[np.nan], value='none')

        temp_train = enc.fit_transform(train[fea].values.reshape(-1,1)).toarray()
        temp_test = enc.transform(test[fea].values.reshape(-1,1)).toarray()

        train_ohe = pd.concat([train_ohe, pd.DataFrame(temp_train, columns=[fea + '_ohe_' + str(enc.categories_[0][i]) for i in range(len(enc.categories_[0]))])], axis=1)
        test_ohe = pd.concat([test_ohe, pd.DataFrame(temp_test, columns=[fea + '_ohe_' + str(enc.categories_[0][i]) for i in range(len(enc.categories_[0]))])], axis=1)
    del train, test
    gc.collect()
    return (train_ohe, test_ohe)

def label_encoding(X_train, X_test):
    train = X_train.copy()
    test = X_test.copy()
    
    train_label = pd.DataFrame()
    test_label = pd.DataFrame()

    for fea in train:
        train[fea] = train[fea].replace(to_replace=[np.nan], value='none')
        test[fea] = test[fea].replace(to_replace=[np.nan], value='none')

        factorised = pd.factorize(train[fea])[1]
        labels = pd.Series(range(len(factorised)), index=factorised)

        temp_train = train[fea].map(labels)
        temp_test = test[fea].map(labels)

        train_label[fea + '_labeled'] = temp_train
        test_label[fea + '_labeled'] = temp_test

    train_label.fillna(-1, inplace=True)
    test_label.fillna(-1, inplace=True)
    del train, test
    gc.collect()
    return (train_label, test_label)

def freq_encoding(X_train, X_test):
    train = X_train.copy()
    test = X_test.copy()
    
    encoded_train_cols = dict()
    encoded_test_cols = dict()
    for col in train:
        train[col] = train[col].replace(to_replace=[np.nan], value='none')
        test[col] = test[col].replace(to_replace=[np.nan], value='none')

        freq_cats = train.groupby([col])[col].count()/train.shape[0]
        encoded_train_cols[str(col) + '_freq'] = train[col].map(freq_cats)
        encoded_test_cols[str(col) + '_freq'] = test[col].map(freq_cats)

    encoded_train_cols = pd.DataFrame(encoded_train_cols)
    encoded_train_cols.fillna(0, inplace=True)
    encoded_test_cols = pd.DataFrame(encoded_test_cols)
    encoded_test_cols.fillna(0, inplace=True)
    del train, test
    gc.collect()
    return (encoded_train_cols, encoded_test_cols)


def mean_encoding(X_train, X_test, target, alpha=0, folds=5, random=True, random_state=913100):
    
    train = pd.concat([X_train, target], axis=1)
    test = X_test.copy()
    encoded_train_cols = dict()
    encoded_test_cols = dict()
    target_mean_gobal = train[target.name].mean()
    
    for col in X_train:
      train[col] = train[col].replace(to_replace=[np.nan], value='none')
      test[col] = test[col].replace(to_replace=[np.nan], value='none')

      # Getting mean for test data
      groups = train.groupby([col])
      nrows_cat = groups[target.name].count()
      target_mean_cats = groups[target.name].mean()
      target_mean_cats_adj = (target_mean_cats*nrows_cat + target_mean_gobal*alpha) / (nrows_cat + alpha) 
      # Mapping mean to test data
      encoded_test_cols[str(col) + '_mean'] = test[col].map(target_mean_cats_adj)

      if folds is None:
        encoded_train_cols[str(col) + '_mean'] = train[col].map(target_mean_cats_adj)
      else:
        kfold = StratifiedKFold(n_splits=folds, shuffle=random, random_state=random_state)  
        parts = []
        # Kfold for train data
        for tr_idx, dev_idx in kfold.split(train.drop(columns=target.name), train[target.name]):
            # Divide data
            base_df, estimate_df = train.iloc[tr_idx], train.iloc[dev_idx]

            # Gettting mean of base_df for estimation
            groups = base_df.groupby([col])
            nrows_cat = groups[target.name].count()
            target_mean_cats = groups[target.name].mean()
            target_mean_cats_adj = (target_mean_cats*nrows_cat + target_mean_gobal*alpha) / (nrows_cat + alpha) 
            # Mapping mran for estimate_df
            parts.extend(estimate_df[col].map(target_mean_cats_adj))

        encoded_train_cols[str(col)+ '_mean'] = parts
  
    encoded_train_cols = pd.DataFrame(encoded_train_cols)
    encoded_train_cols.fillna(target_mean_gobal, inplace=True)
    encoded_test_cols = pd.DataFrame(encoded_test_cols)
    encoded_test_cols.fillna(target_mean_gobal, inplace=True)
    del train, test
    gc.collect() 
    return (encoded_train_cols, encoded_test_cols)

def scoring_ngboost_clf(X_train, y_train, X_dev, y_dev, random_state=913100, verbose=False):
    iterations = []
    train_scores = []
    dev_scores = []
  
  
    log_iters = list(set((np.logspace(math.log(1, 8), math.log(500, 8), 
                                        num=50, endpoint=True, base=8, 
                                        dtype=np.int))))
    for estimators in sorted(log_iters):
        model = NGBoost(n_estimators=estimators, random_state=random_state)
        model.fit(X_train, y_train)
        y_train_pred_scores = model.predict_proba(X_train)
        y_dev_pred_scores = model.predict_proba(X_dev)

        train_scores.append(roc_auc_score(y_train, y_train_pred_scores[:, 1]))
        dev_scores.append(roc_auc_score(y_dev, y_dev_pred_scores[:, 1]))
        iterations.append(estimators)
        if verbose:
            print(f'{iterations[-1]}/{len(log_iters)}', train_scores[-1], dev_scores[-1])
  
    best_score = max(dev_scores)
    best_iter = iterations[dev_scores.index(best_score)]
    if verbose:
        print(f'Best score: {best_score}. Best iter: {best_iter}')
    return (train_scores, dev_scores, iterations, model)

def test_all_encodings(train, dev, target_name):
    # Format: encoding function, encoding params, encoding name, encoding color
    encoding_settings = [
                      [one_hot_encoding, {}, 'One hot encoding', '#E7E005'],
                      [label_encoding, {}, 'Label encoding', '#960000'],
                      [freq_encoding, {}, 'Frequency encoding', '#FF2F02'],
                      [mean_encoding, {'alpha':0, 'folds':None, 'target':train['label']}, 'Mean encoding, alpha=0', '#A4C400'],
                      [mean_encoding, {'alpha':2, 'folds':None, 'target':train['label']}, 'Mean encoding, alpha=2', '#73B100'],
                      [mean_encoding, {'alpha':5, 'folds':None, 'target':train['label']}, 'Mean encoding, alpha=5', '#2B8E00'],
                      [mean_encoding, {'alpha':5, 'folds':3, 'target':train['label']}, 'Mean encoding, alpha=5, 3 folds', '#00F5F2'],
                      [mean_encoding, {'alpha':5, 'folds':5, 'target':train['label']}, 'Mean encoding, alpha=5, 5 folds', '#00BAD3'],
    ]
    scoring_func = scoring_ngboost_clf
    plt.figure(figsize=(10,7))

    review_rows = []

    for encoding_func, encoding_params, str_name, color in encoding_settings:
        print(str_name)
        X_train, X_dev = encoding_func(train.drop(columns=target_name), dev.drop(columns=target_name), **encoding_params)

        scores = scoring_func(X_train, train[target_name], X_dev, dev[target_name])

        train_scores, dev_scores, iters, _ = scores
        plt.plot(iters,  dev_scores,  label='Test, ' + str_name, linewidth=1.5, color=color)

        best_score_dev = max(dev_scores)
        best_iter_dev = iters[dev_scores.index(best_score_dev)]
        best_score_train = max(train_scores[:best_iter_dev])

        print(f'Best score for {str_name} is {best_score_dev}, on estimators {best_iter_dev}')
        review_rows.append([str_name, best_score_train, best_score_dev, best_iter_dev])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    columns = ['Encoding', 'Train AUC score on best iteration', 'Best AUC score (test)', 'Best iteration (test)']
    return pd.DataFrame(review_rows, columns=columns)

