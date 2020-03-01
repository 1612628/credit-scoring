import numpy as np
from sklearn import base
from sklearn.model_selection import KFold
import pandas as pd
import gc

# DataFilter class for filter opration on data
class cs_data_filter(object):
  """ DataFilter
  
  Used for filter opration on data
  
  Parameters
  ----------
  
  Attributes
  ----------
  """
  @staticmethod
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
      miss_values = cs_data_filter._missValueTable(train_data,['None','na','Nan','NaN'])
      drop_features = miss_values[miss_values[miss_values.columns[1]] >= missing_rate].index.values.tolist()
      train_data_cp = train_data.drop(columns = drop_features).copy()
      test_data_cp = test_data.drop(columns = drop_features).copy()
      return train_data_cp, test_data_cp
  @staticmethod
  def ReplaceValues(data, patterns, to_value):
      """ReplaceValues
      This function used to replace values with pattern to to_value value
      Parameters
      ----------
      data: Series (required)
        The Series need to replace value
      patterns: list of patterns (required)
        The list of corresponding patterns need to be replaced
      to_value: object (required)
        Value used for replacing
      Return
      ----------
      data: Series
        The replaced-value Series
      """
      # Validate variables information
      assert (type(data) == pd.Series),"data must be pandas Series"
      assert (type(patterns) == list),"patterns must be list"
      assert (to_value is not None), "to_value must be specified"
      data_cp = data.copy()
      for indx,value in enumerate(data_cp):
        if cs_data_filter._isPattern(value,patterns):
          data_cp.iat[indx] = to_value
      return data_cp
  @staticmethod
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
  @staticmethod
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
      return cs_data_filter._isPattern(value, missing_patterns) or (type(value) == float and np.isnan(value)) 
  @staticmethod
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
        sr = pd.Series({col_name:df[df.apply(lambda row: cs_data_filter._isMissing(row[col_name],missing_patterns), axis=1)].shape[0]})
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
        print(f'Your selected dataframe has {df.shape[1]} columns.\nThere are {miss_values_table.shape[0]} columns that have missing values');
      return miss_values_table
  @staticmethod
  def PresentCompleteCaseAnalysis(data):
      """
      This function is to present all complete case analysis
      Parameters
      ----------
      data: DataFrame (required)
        The data with missing values
      Return
      ----------
      cpl_data: DataFrame
        The dataframe contains all complete cases
      """
      #Validate variables
      assert (type(data) == pd.DataFrame),'data must be pandas DataFrame'
      data_miss = data.isnull()
      idx_good = []
      for idx, row in data_miss.iterrows():
        if not row.any():
          idx_good.append(idx)
      data_cpl = data.iloc[idx_good,:].copy().reset_index()
      gc.enable()
      del data_miss
      gc.collect()
      return data_cpl
  @staticmethod
  def high_corr(data,threshold=0.8):
    corr = data.corr()

    high_corr_dic = {col:[] for col in list(corr.columns)}
    for i in corr.index:
      for j in corr.loc[i,:].index:
        if (i != j) and (corr.loc[i,j] >= 0.8) and (len(high_corr_dic[j]) == 0):
          high_corr_dic[i].append(j)
    return high_corr_dic
  @staticmethod
  def drop_high_corr(train_data, test_data, threshold=0.8):

    train_drop = train_data.copy()
    to_drop = []
    while True:
      high_corr_dic = cs_data_filter.high_corr(train_drop, threshold)
      high_corr_dic = sorted(high_corr_dic.items(), key=lambda x:len(x[1]), reverse=True)
      if (len(high_corr_dic) <= 0) or (len(high_corr_dic[0][1]) <= 0):
        break
      to_drop.append(high_corr_dic[0][0])
      train_drop = train_drop.drop(columns=[high_corr_dic[0][0]])

    test_drop = test_data.drop(columns=to_drop)
    return train_drop, test_drop
