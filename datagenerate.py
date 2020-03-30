import pandas as pd
from pandas.core import algorithms
import numpy as np
from scipy.stats import spearmanr
import gc
gc.enable()

class cs_data_generate(object):

    @staticmethod
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

class WOEBINS:
  """
  Calculating Weight of Evidence and Information Value for input data
  """

  def __init__(self, corr_threshold=0.8, max_bins=20, force_bins=3):
    self.corr_threshold_ = corr_threshold
    self.max_bins_ = max_bins
    self.force_bins_ = force_bins

  def _numeric_bins(self, x, y):
    """
    Calculating WOE for input feature x
    
    Parameters
    ----------
    x: 1-d array-like [pandas Series, numpy ndarray] (required)
      The numeric input feature
    y: 1-d array-like [pandas Series, numpy ndarray] (required)
      The target feature
    
    Return
    ----------
    out: DataFrame 
      The dataframe Weight of Evidence and Information Values
    """

    df = pd.DataFrame({'X': x, 'Y': y})
    ismiss = df[df.isnull()]
    notmiss = df[df.notnull()]
    r = 0
    n = self.max_bins_
    #Calculating bins for input feature x
    while np.abs(r) < self.corr_threshold_:
      try:
        d1 = pd.DataFrame({'X': notmiss.X, 'Y':notmiss.Y, 'Bucket': pd.qcut(notmiss.X, n)})
        d2 = d1.groupby(['Bucket'], as_index=True)
        r, _ = spearmanr(d2.mean().X, d2.mean().Y)
        n -= 1 
      except:
        n -= 1 

    # If feature x has been binned in just only 1 bin, we have to re-bin
    if len(d2) <= 1:
      n = self.force_bins_

      d1 = pd.DataFrame({'X': notmiss.X, 'Y':notmiss.Y, 'Bucket': pd.qcut(notmiss.X, n, duplicates='drop')}) 
      d2 = d1.groupby(['Bucket'], as_index=True)

    # Create WOE and IV
    d3 = pd.DataFrame()
    d3['MIN_VALUE'] = d2.min().X
    d3['MAX_VALUE'] = d2.max().X
    d3['COUNT'] = d2.count().Y
    d3['DEFAULT'] = d2.sum().Y
    d3['NONDEFAULT'] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop=True)

    if len(ismiss.index) > 0:
      d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
      d4['MAX_VALUE'] = np.nan 
      d4['COUNT'] = ismiss.count().Y
      d4['DEFAULT'] = ismiss.sum().Y
      d4['NONDEFAULT'] = ismiss.count().Y - ismiss.sum().Y
      d3 = d3.append(d4, ignore_index=True)
    
    d3['DEFAULT_RATE'] = d3['DEFAULT']/d3['COUNT']
    d3['NONDEFAULT_RATE'] = d3['NONDEFAULT']/d3['COUNT']
    d3['DIST_DEFAULT'] = d3['DEFAULT']/d3.sum()['DEFAULT']
    d3['NONDIST_DEFAULT'] = d3['NONDEFAULT']/d3.sum()['NONDEFAULT']
    d3['WOE'] = np.log(d3['NONDIST_DEFAULT'] / d3['DIST_DEFAULT'])
    d3['IV'] = (d3['NONDIST_DEFAULT'] - d3['DIST_DEFAULT']) * np.log(d3['NONDIST_DEFAULT'] / d3['DIST_DEFAULT'])
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3['IV'] = d3['IV'].sum()

    del ismiss, notmiss, df, d1, d2, d4
    gc.collect()
    return d3 
  
  def _category_bins(self, x, y):
    """
    Calculating WOE for input categorical feature x
    
    Parameters
    ----------
    x: 1-d array-like [pandas Series, numpy ndarray] (required)
      The numeric input feature
    y: 1-d array-like [pandas Series, numpy ndarray] (required)
      The target feature
    
    Return
    ----------
    out: DataFrame 
      The dataframe Weight of Evidence and Information Values
    """

    df = pd.DataFrame({"X":x, "Y":y})
    ismiss = df[df.isnull()]
    notmiss = df[df.notnull()]

    d2 = notmiss.groupby('X', as_index=True)

    d3 = pd.DataFrame()
    d3['COUNT'] = d2.count().Y
    d3['MIN_VALUE'] = d2.sum().Y.index
    d3['MAX_VALUE'] = d3['MIN_VALUE']
    d3['DEFAULT'] = d2.sum().Y
    d3['NONDEFAULT'] = d2.count().Y - d2.sum().Y

    if len(ismiss.index) > 0:
      d4 = pd.DataFrame()
      d4['MAX_VALUE'] = np.nan
      d4['COUNT'] = ismiss.count().Y
      d4['DEFAULT'] = ismiss.sum().Y
      d4['NONDEFAULT'] = ismiss.count().Y - ismiss.sum().Y
      d3 = d3.append(d4, ignore_index=True)
    
    d3['DEFAULT_RATE'] = d3['DEFAULT'] / d3['COUNT']
    d3['NONDEFAULT_RATE'] = d3['NONDEFAULT'] / d3['COUNT']
    d3['DIST_DEFAULT'] = d3['DEFAULT'] / d2.sum().Y
    d3['NONDIST_DEFAULT'] = d3['NONDEFAULT'] / d2.sum().Y
    d3['WOE'] = np.log(d3['NONDIST_DEFAULT'] / d3['DIST_DEFAULT'])
    d3['IV'] = (d3['NONDIST_DEFAULT'] - d3['DIST_DEFAULT']) * np.log(d3['NONDIST_DEFAULT'] / d3['DIST_DEFAULT'])
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3['IV'] = d3['IV'].sum()
    
    del df, ismiss, notmiss, d2, d4
    return d3 


  def fit(self, data, target):
    return self 
    

  def transform(self, data, target):
    """
    Calculating Weight of Evidence and Information Value for input data
    
    Parameters
    ----------
    data: DataFrame (n samples x d features) (required)
      The input data 
    target: 1-d array-like [pandas Series, numpy ndarray] (required)
    
    Return
    ----------
    out: DataFrame 
      The dataframe Weight of Evidence and Information Values
    """
    iv_df = pd.DataFrame()    

    for col in data:
      if np.issubdtype(data[col], np.number) and len(data[col].unique()) > 2:
        temp = self._numeric_bins(data[col], target)
      else:
        temp = self._category_bins(data[col], target)
      
      temp['VARNAME'] = col
      iv_df = iv_df.append(temp, ignore_index=True)
      
      del temp
      gc.collect()

    iv = pd.DataFrame({'IV': iv_df.groupby(['VARNAME'])['IV'].max()})
    iv = iv.reset_index(drop=True) 
    return (iv_df, iv)
  
  def fit_transform(self, data, target):
    return self.fit(data, target).transform(data, target)

