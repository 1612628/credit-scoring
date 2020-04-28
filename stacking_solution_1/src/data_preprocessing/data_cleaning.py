# data_cleaning
import numpy as np
import pandas as pd
from ..common.utils import get_logger
from .base import DropFeaturesByMissingRate, GenerateIsMissFeatures, drop_high_corr, replaceAll

logger = get_logger()
import gc
gc.enable()
from unidecode import unidecode
from ast import literal_eval

from feature_engine import categorical_encoders as ce
import scorecardpy as sc


class KalapaCleaning:
    """
    Cleaning kalapa dataset
    """
    def __init__(self):
        logger.info('KalapaPreprocessing...')
    
    def fit_transform(self, train ,test):
        return self.fit(train, test).transform(train, test)
        
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
        
        logger.info('KalapaPreprocessing, transform, rename')
        train, test = self._rename(train, test)
        
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
        train, test = self._numeric_transform(train, test)
        
        train.set_index("id",inplace=True, drop=True)
        test.set_index("id",inplace=True, drop=True)
        for col in train:
            if train[col].dtype != 'object':
              train[col] = train[col].astype('float32')

        for col in test:
          if test[col].dtype != 'object':
            test[col] = test[col].astype('float32')

        
        return (train, test)        
    def _rename(self, train, test):
        short_column_names = dict([(f'FIELD_{i}', str(i)) for i in range(1, 58)])
        return train.rename(columns=short_column_names), test.rename(columns=short_column_names)
    
    def _categorical_cleaning(self, train, test, *args, **kwargs):
        # Data cleaning
        replace_dict = {" ":"", "y":"i", ".":"", "-":""} 
        train['province'].fillna('unk_province', inplace=True)
        train['province'] = train['province'].apply(self._clean_province)
        test['province'].fillna('unk_province', inplace=True)
        test['province'] = test['province'].apply(self._clean_province)

        train['district'].fillna('unk_district', inplace=True)
        train['district'] = train['district'].apply(self._clean_district)
        test['district'].fillna('unk_district', inplace=True)
        test['district'] = test['district'].apply(self._clean_district)
        
        train['maCv'].fillna('unk_macv', inplace=True)
        train['maCv'] = train['maCv'].apply(self._clean_macv)
        test['maCv'].fillna('unk_macv', inplace=True)
        test['maCv'] = test['maCv'].apply(self._clean_macv)
        
        train['7'].fillna('[]', inplace=True)
        train['7'] = train['7'].apply(literal_eval)
        test['7'].fillna('[]', inplace=True)
        test['7'] = test['7'].apply(literal_eval)
        
        train['8'].fillna('unk_8', inplace=True)
        test['8'].fillna('unk_8', inplace=True)

        train['9'] = train['9'].apply(self._clean_9)
        test['9'] = test['9'].apply(self._clean_9)

        train['10'].fillna('unk_10', inplace=True)
        train['10'] = train['10'].apply(self._clean_10)
        test['10'].fillna('unk_10', inplace=True)
        test['10'] = test['10'].apply(self._clean_10)
                
        train['13'].fillna('unk_13', inplace=True)
        test['13'].fillna('unk_13', inplace=True)
        value_not_existed_train_13 = list(set(test['13'].unique()) - set(train['13'].unique()))
        test['13'].replace(to_replace=value_not_existed_train_13, value='unk_13', inplace=True)
        
        train['17'].fillna('unk_17', inplace=True)
        train['17'] = train['17'].apply(self._clean_17)
        test['17'].fillna('unk_17', inplace=True)
        test['17'] = test['17'].apply(self._clean_17)
        
        train['18'].fillna('unk_18', inplace=True)
        test['18'].fillna('unk_18', inplace=True)
        
        train['19'].fillna('unk_19', inplace=True)
        test['19'].fillna('unk_19', inplace=True)
        
        train['20'].fillna('unk_20', inplace=True)
        test['20'].fillna('unk_20', inplace=True)
        
        train['23'].fillna('unk_23', inplace=True)
        test['23'].fillna('unk_23', inplace=True)

        train['24'].fillna('unk_24', inplace=True)
        train['24'] = train['24'].apply(self._clean_24)
        test['24'].fillna('unk_24', inplace=True)
        test['24'] = test['24'].apply(self._clean_24)
        
        train['25'].fillna('unk_25', inplace=True)
        test['25'].fillna('unk_25', inplace=True)
        
        train['26'].fillna('unk_26', inplace=True)
        test['26'].fillna('unk_26', inplace=True)
        
        train['27'].fillna('unk_27', inplace=True)
        test['27'].fillna('unk_27', inplace=True)
        
        train['28'].fillna('unk_28', inplace=True)
        test['28'].fillna('unk_28', inplace=True)
        
        train['29'].fillna('unk_29', inplace=True)
        train['29'].replace('None', 'unk_29', inplace=True)
        test['29'].fillna('unk_29', inplace=True)
        test['29'].replace('None', 'unk_29', inplace=True)
        
        train['30'].fillna('unk_30', inplace=True)
        train['30'].replace('None', 'unk_30', inplace=True)
        test['30'].fillna('unk_30', inplace=True)
        test['30'].replace('None', 'unk_30', inplace=True)
        
        train['31'].fillna('unk_31', inplace=True)
        train['31'].replace('None', 'unk_31', inplace=True)
        test['31'].fillna('unk_31', inplace=True)
        test['31'].replace('None', 'unk_31', inplace=True)

        train['35'].fillna('unk_35', inplace=True)
        test['35'].fillna('unk_35', inplace=True)
        
        train['36'].fillna('unk_36', inplace=True)
        train['36'].replace('None', 'unk_36', inplace=True)
        test['36'].fillna('unk_36', inplace=True)
        test['36'].replace('None', 'unk_36', inplace=True)
        
        train['37'].fillna('unk_37', inplace=True)
        train['37'].replace('None', 'unk_37', inplace=True)
        test['37'].fillna('unk_37', inplace=True)
        test['37'].replace('None', 'unk_37', inplace=True)
        
        train['38'].fillna('unk_38', inplace=True)
        train['38'].replace('None', 'unk_38', inplace=True)
        test['38'].fillna('unk_38', inplace=True)
        test['38'].replace('None', 'unk_38', inplace=True)
        
        train['39'].fillna('unk_39', inplace=True)
        train['39'] = train['39'].apply(self._clean_39)
        test['39'].fillna('unk_39', inplace=True)
        test['39'] = test['39'].apply(self._clean_39)
        value_not_existed_train_39 = list(set(test['39'].unique()) - set(train['39'].unique()))
        test['39'].replace(to_replace=value_not_existed_train_39, value='unk_39', inplace=True)
        
        train['41'].fillna('unk_41', inplace=True)
        train['41'] = train['41'].apply(self._clean_41)
        test['41'].fillna('unk_41', inplace=True)
        test['41'] = test['41'].apply(self._clean_41)
        
        train['42'].fillna('unk_42', inplace=True)
        train['42'] = train['42'].apply(self._clean_42)
        test['42'].fillna('unk_42', inplace=True)
        test['42'] = test['42'].apply(self._clean_42)
        
        train['43'].fillna('unk_43', inplace=True)
        train['43'] = train['43'].apply(self._clean_43)
        test['43'].fillna('unk_43', inplace=True)
        test['43'] = test['43'].apply(self._clean_43)
        
        train['44'].fillna('unk_44', inplace=True)
        train['44'] = train['44'].apply(self._clean_44)
        test['44'].fillna('unk_44', inplace=True)
        test['44'] = test['44'].apply(self._clean_44)
        
        return (train, test)
    
    def _categorical_ge(self, train, test):
        train['district'] = train['province'] + ' - ' + train['district']
        test['district'] = test['province'] + ' - ' + test['district']
        
        train['7_count'] = self._ge_7_count(train['7'])
        test['7_count'] = self._ge_7_count(train['7'])
        elements_7 = set(train['7'].unique()).intersection(set(test['7'].unique()))
        for ele in elements_7:
            train[f'7_count_{ele}'] = train['7'].apply(lambda x: x.count(ele))
            test[f'7_count_{ele}'] = test['7'].apply(lambda x: x.count(ele))      
             
        return (train, test)

    def _categorical_fe(self, train, test):
        train['35'] = train['35'].apply(self._fe_35)
        test['35'] = test['35'].apply(self._fe_35)
        test['35'].fillna(-1, inplace=True)

        target_enc_cols = 'province district maCv 8 9 10 13 17 18 19 20 23 24 25 26 27 28 29 30 31 36 37 38 39 41 42 43 44 45 47 48 49'.split() 
        target_encoder = ce.MeanCategoricalEncoder(variables=target_enc_cols)
        train_cat_encoded = pd.DataFrame(target_encoder.fit_transform(train[target_enc_cols], train['label']), columns=target_enc_cols).add_suffix('_te')
        test_cat_encoded = pd.DataFrame(target_encoder.transform(test[target_enc_cols]), columns=target_enc_cols).add_suffix('_te')
        
        test_cat_encoded.fillna(-1, inplace=True)
        
        train = pd.concat([train, train_cat_encoded], axis=1)
        test = pd.concat([test, test_cat_encoded], axis=1)  
         
        
        return (train, test) 
    
    def _categorical_transform(self, train, test):
        
        train, test = self._categorical_cleaning(train, test)
        train, test = self._categorical_ge(train, test)
        train, test = self._categorical_fe(train, test)
        
        return (train, test) 
     
    def _numeric_cleaning(self, train, test, *args, **kwargs):
        
        train['11'].replace(['None'], np.nan, inplace=True)
        test['11'].replace(['None'], np.nan, inplace=True)
        
        f12_outlier_value = set(train['12'].unique).union(set(test[12].unique()))
        f12_outlier_value.discard(0)
        f12_outlier_value.discard(1)
        train['12'].replace(list(f12_outlier_value), np.nan, inplace=True)
        test['12'].replace(list(f12_outlier_value), np.nan, inplace=True)

        f40_mapper={1:1, 2:2, 3:3, 4:4, 6:6}
        train['40'] = train['40'].mapper(f40_mapper)
        test['40'] = test['40'].map(f40_mapper)
        
        train['45'].replace(['None'], np.nan, inplace=True)
        test['45'].replace(['None'], np.nan, inplace=True)
        
        for col in '54 55 56 57'.split():
            train[col].replace(['nan'], np.nan, inplace=True)
            test[col].replace(['nan'], np.nan, inplace=True)
   
        return (train, test)
    
    def _numeric_ge(self, train, test):
        train['age'] = train['age_source1'].mask(train['age_source1'].isnull(), train['age_source2'])
        test['age'] = test['age_source1'].mask(test['age_source1'].isnull(), test['age_source2'])
        
        return train, test
    
    def _numeric_fe(self, train, test):
         
        return (train, test)
    
    def _numeric_transform(self, train, test):
        train, test = self._numeric_cleaning(train, test)
        train, test = self._numeric_ge(train, test)
        train, test = self._numeric_fe(train, test)
        
        return (train, test)
    
    # Categorical data
    def _clean_province(self, value):
        replace_dict = {" ":"", "y":"i", ".":"", "-":""}
        return replaceAll(unidecode(str.lower(value)), replace_dict)
    
    def _clean_district(self, value):
        replace_dict = {" ":"", "y":"i", ".":"", "-":""}
        return replaceAll(unidecode(str.lower(value)), replace_dict)
    
    def _clean_macv(self, value):
        replace_dict = {" ":"", "y":"i", ".":"", "-":"", 'None': 'unk_macv'}
        return replaceAll(unidecode(str.lower(value)), replace_dict)
   
    def _clean_7(self, value):
        if pd.isna(value):
            return '[]'
        return value
    
    def _clean_9(self, value):
        if value == 'na':
            return 'unk_9'
        return value
     
    def _clean_10(self, value):
        if value == 'None':
            value = 'unk_10'
        return str.lower(value)
        
    def _clean_17(self, value):
        if value == 'None':
            return 'unk_17'
        return value
    
    def _clean_24(self, value):
        if value == 'None':
            return 'unk_24'
        return value
    
    def _clean_39(self, value):
        if value == 'None':
            return 'unk_39'
        return value
    
    def _clean_41(self, value):
        if value == 'None':
            return 'unk_41'
        return value
    
    def _clean_42(self, value):
        if value == 'None':
            return 'unk_42'
        return value
    
    def _clean_43(self, value):
        if value == 'None':
            return 'unk_43'
        return value
    
    def _clean_44(self, value):
        if value == 'None':
            return 'unk_44'
        return value

    def _ge_7_count(self, col):
       return col.apply(len) 
    
    def _fe_35(self, value):
        mapper = {'unk_35':-1, 'Zero': 0, 'One':1, 'Two':2, 'Three':3, 'Four': 4}
        return mapper[value]
   
    def _fe_41(self, value):
        mapper = {'unk_41':-1, 'I':1, 'II':2, 'III':3, 'IV':4, 'V':5}
        return mapper[value]

    # Numerical data
    
    def _fe_3(self, value):
        if value == 0 or value == -1:
            return 0
        for i in range(1, 23):
            t = int(i*365.25) - 13
            if t - 30 <= value <= t + 30:
                return min(12, i)
        return -999
