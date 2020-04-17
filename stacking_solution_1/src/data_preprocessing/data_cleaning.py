# data_cleaning
import numpy as np
import pandas as pd
from ..common.utils import get_logger
from .base import DropFeaturesByMissingRate, GenerateIsMissFeatures, drop_high_corr, replaceAll

logger = get_logger()
import gc
gc.enable()
from unidecode import unidecode

class KalapaCleaning:
    """
    Cleaning kalapa dataset
    """
    def __init__(self):
        logger.info('KalapaPreprocessing...')
        
    def fit(self, X_train, X_test, *args, **kwargs):
        logger.info('KalapaPreprocessing, fit')
        return self
    
    def transform(self, X_train, X_test, *args, **kwargs):
        logger.info('KalapaPreprocessing, transform')
        
        logger.info('KalapaPreprocessing, transform, replace missing value with np.nan')
        train = X_train.copy()
        train = X_train.replace(['Nan','na','None','NaN','[]',-1], np.nan)
        test = X_test.copy()
        test = X_test.replace(['Nan','na','None','NaN','[]',-1], np.nan)
        
        logger.info('KalapaPreprocessing, transform, drop features with high missing rate')
        train, test = DropFeaturesByMissingRate(train_data=train, test_data=test, missing_rate=0.8)
        
        logger.info('KalapaPreprocessing, transform, generate is missing feature on each original feature')
        ismiss_train = GenerateIsMissFeatures(train.drop(columns=['id','label']))
        ismiss_test = GenerateIsMissFeatures(test.drop(columns=['id']))
        only_one_value = []
        for fea in ismiss_train:
            if len(ismiss_train[fea].unique()) <= 1:
                only_one_value.append(fea)
        ismiss_train = ismiss_train.drop(columns=only_one_value)
        ismiss_test = ismiss_test.drop(columns=only_one_value)
        ismiss_train, ismiss_test= drop_high_corr(ismiss_train, ismiss_test) 
        
        logger.info('KalapaPreprocessing, transform, concat new feature with train set and test set')
        ismiss_train = GenerateIsMissFeatures(train.drop(columns=['id','label']))
        train = pd.concat([train, ismiss_train], axis=1)
        test = pd.concat([test, ismiss_test], axis=1)

        gc.enable()
        del ismiss_train, ismiss_test 
        gc.collect() 
        
        logger.info('KalapaPreprocessing, transform, add incomplete feature: count number of misisng values for each sample')
        train['incomplete'] = train.isnull().sum(axis=1)
        test['incomplete'] = test.isnull().sum(axis=1)
        
        train, test = self._categorical_transform(train, test)
        train, test = self._boolean_transform(train, test)
        train, test = self._numeric_transform(train, test)
        
        train.set_index("id",inplace=True, drop=True)
        test.set_index("id",inplace=True, drop=True)
        for col in train:
            if train[col].dtype != 'object':
              train[col] = train[col].astype('float32')
            elif train[col].dtype == 'object':
              train[col] = train[col].replace([np.nan],'none')

        for col in test:
          if test[col].dtype != 'object':
            test[col] = test[col].astype('float32')
          elif train[col].dtype == 'object':
            test[col] = test[col].replace([np.nan],'none')

        train.drop(columns=['FIELD_7'], inplace=True)
        test.drop(columns=['FIELD_7'], inplace=True)
        
        
        
    def _categorical_transform(self, train, test, *args, **kwargs):
        replace_dict = {" ":"", "y":"i", ".":"", "-":""}
        
        for fea in ['province', 'district', 'maCv']:
            train[fea].replace(to_replace=[np.nan], value='none', inplace=True)
            train[fea] = train[fea].apply(unidecode).apply(str.lower).apply(lambda text: replaceAll(text,replace_dict))
            train[fea].replace(['none'], np.nan, inplace=True)
            test[fea].replace(to_replace=[np.nan], value='none', inplace=True)
            test[fea] = test[fea].apply(unidecode).apply(str.lower).apply(lambda text: replaceAll(text,replace_dict))
            test[fea].replace(['none'], np.nan, inplace=True)

        for feature in ['FIELD_10','FIELD_13','FIELD_39']:
            train[feature].replace(to_replace=[np.nan], value='none', inplace=True)
            train[feature] = train[feature].apply(unidecode).apply(str.lower)
            train[feature].replace(['none'], np.nan, inplace=True)
            test[feature].replace(to_replace=[np.nan], value='none', inplace=True)
            test[feature] = test[feature].apply(unidecode).apply(str.lower)
            test[feature].replace(['none'], np.nan, inplace=True)
        return (train, test)
    
    def _boolean_transform(self, train, test):
        bool_features = ['FIELD_1', 'FIELD_2', 'FIELD_12', 'FIELD_14', 'FIELD_15', 
                         'FIELD_18', 'FIELD_19', 'FIELD_20', 'FIELD_23', 'FIELD_25', 
                         'FIELD_26', 'FIELD_27', 'FIELD_28', 'FIELD_29', 'FIELD_30', 
                         'FIELD_31', 'FIELD_32', 'FIELD_33', 'FIELD_34', 'FIELD_36', 
                         'FIELD_37', 'FIELD_38', 'FIELD_46', 'FIELD_47', 'FIELD_48', 'FIELD_49']
        mapper = {'0':0, '1':1, 1.0:1, 0.0:0, 1:1, 0:0, 'false':0, 'true':1, False:0, True:1, 'FALSE':0,'TRUE':1}
        
        for feature in bool_features:
            train[feature] = train[feature].map(mapper)
            test[feature] = test[feature].map(mapper)

        # Transform FIELD_12. Convert number in str to float,transforms 'None', HT','TN','DK', 'GD', 'XK', 'DN', 'DT', to np.nan. 
        # Then impute np.nan using most_frequency SimpleImputer
        # Transforms 'HT','TN', 'None' to np.nan
        train['FIELD_12'] = train['FIELD_12'].replace(['None','HT','TN'], np.nan)
        # Transforms 'None','DK', 'GD', 'XK', 'DN', 'DT', 'HT' to np.nan
        test['FIELD_12'] = test['FIELD_12'].replace(['None','DK', 'GD', 'XK', 'DN', 'DT', 'HT'], np.nan)
        # Convert number in str to float
        train['FIELD_12'] = pd.to_numeric(train['FIELD_12'],errors='coerce')
        test['FIELD_12'] = pd.to_numeric(train['FIELD_12'],errors='coerce')
        # Drop FIELD_23, FIELD_31 because containing only 1 unique values 
        train.drop(columns=['FIELD_23', 'FIELD_31'], inplace=True)
        test.drop(columns=['FIELD_23', 'FIELD_31'], inplace=True)
        return (train, test)
    
    def _numeric_transform(self, train, test, *args, **kwargs):
        # Transform age_source1, age_source2
        # age1_median = train[train['age_source1'] != 'None']['age_source1'].median()
        # age2_median = train[train['age_source2'] != 'None']['age_source2'].median()

        # Train set
        train.loc[train['age_source1'] < 18,'age_source1'] = np.nan
        train.loc[train['age_source2'] < 18,'age_source2'] = np.nan

        age_se = pd.Series([])
        for age1, age2 in zip(train['age_source1'],train['age_source2']):
            if (age1 != np.nan) and (age2 != np.nan):
                if (age1 == age2):
                    age_se = age_se.append(pd.Series(age1), ignore_index=True)
                else:
                    age_se=age_se.append(pd.Series((age1+age2)/2), ignore_index=True)
            elif (age1 == np.nan) and (age2 == np.nan):
                age_se=age_se.append(pd.Series(np.nan), ignore_index=True)
            elif age1 != np.nan:
                age_se=age_se.append(pd.Series(age1), ignore_index=True)
            else:
                age_se=age_se.append(pd.Series(age2), ignore_index=True)
        train['age'] = age_se
        train = train.drop(columns=['age_source1','age_source2'])
        train.loc[train['age'] < 18,'age'] = np.nan
       
        # Test set
        test.loc[test['age_source1'] < 18,'age_source1'] = np.nan
        test.loc[test['age_source2'] < 18,'age_source2'] = np.nan

        age_se = pd.Series([])
        for age1, age2 in zip(test['age_source1'],test['age_source2']):
            if (age1 != np.nan) and (age2 != np.nan):
                if (age1 == age2):
                    age_se = age_se.append(pd.Series(age1), ignore_index=True)
                else:
                    age_se=age_se.append(pd.Series((age1+age2)/2), ignore_index=True)
            elif (age1 == np.nan) and (age2 == np.nan):
                age_se=age_se.append(pd.Series(np.nan), ignore_index=True)
            elif age1 != np.na:
                age_se=age_se.append(pd.Series(age1), ignore_index=True)
            else:
                age_se=age_se.append(pd.Series(age2), ignore_index=True)
        test['age'] = age_se
        test = test.drop(columns=['age_source1','age_source2'])
        test.loc[test['age'] < 18,'age'] = np.nan
        
        del age_se
        gc.collect()
        
        # Transform FIELD_3. -1 to np.nan
        train.loc[train['FIELD_3'] < 0, 'FIELD_3']= np.nan
        test.loc[test['FIELD_3'] < 0, 'FIELD_3']= np.nan
        # Transform FIELD_11, FIELD_45. Convert number in str to float, NaN using KNN
        for feature in ['FIELD_11', 'FIELD_45']:
            train[feature] = train[feature].replace(['None'], np.nan)
            train[feature] = pd.to_numeric(train[feature],errors='coerce')

            test[feature] = test[feature].replace(['None'], np.nan)
            test[feature] = pd.to_numeric(test[feature],errors='coerce')
        
        for feature_name in list(train.columns):
            train[feature_name] = train[feature_name].replace(['None'], np.nan)
        for feature_name in list(test.columns):
            test[feature_name] = test[feature_name].replace(['None'], np.nan) 
        return (train, test) 
        
    

        