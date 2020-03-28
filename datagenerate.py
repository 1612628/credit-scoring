import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

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

class KMeansFeaturizer:
  """
  Transform numeric data into k-mean cluster membership

  This transformation run kmean on input data in convert each data point into the index of 
  the closest cluster. If target information is given, then it is scaled and included as input 
  of k-means in order to derive clusters that obey the classification as well as as group of similar point together
  """
  def __init__(self, k=100, target_scale=2.0, random_state=None):
    self.k_ = k
    self.target_scale_ = target_scale
    self.random_state_ = random_state
  
  def convert_to_numpy(self, X):
    if type(X) == pd.DataFrame:
      try:
        return X.to_numpy()
      except:
        ValueError('X must be pandas DataFrame or numpy ndarray')
    return X

  def fit(self, X, y=None, *args, **kwargs):
    """
    Run k-means on the input data and find centroids.
    """
    X = self.convert_to_numpy(X)
    if y is None:
      # No target information, just do plain k-means
      km_model = KMeans(n_clusters=self.k_, random_state=self.random_state_, n_init=20, **kwargs)

      km_model.fit(X)

      self.cluster_centers_ = km_model.cluster_centers_
      self.km_model_ = km_model
      return self

    # With target information
    data = np.hstack((X, y[:, np.newaxis]*self.target_scale_)) 

    km_model_pretrain = KMeans(n_clusters=self.k_, random_state=self.random_state_, n_init=20, **kwargs)
    km_model_pretrain.fit(data)

    km_model = KMeans(n_clusters=self.k_, init=km_model_pretrain.cluster_centers_[:,:-2], n_init=20, random_state=self.random_state_, **kwargs)
    km_model.fit(X)
    self.cluster_centers_ = km_model.cluster_centers_
    self.km_model_ = km_model
    return self
  
  def transform(self, X, **kwargs):
    cluster = self.km_model_.transform(X)
    return cluster[:, np.newaxis]
  
  def fit_transform(self, X, y=None, **kwargs):
    self.fit(X,y,**kwargs).transform(X)