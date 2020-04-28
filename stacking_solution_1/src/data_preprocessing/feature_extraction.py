# feature extraction
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn import base
from sklearn.base import BaseEstimator, ClassifierMixin
from ..common.utils import get_logger
import scipy

import gc
gc.enable()

logger = get_logger()

import scorecardpy as sc
from pylab import rcParams
from .base import mean_encoding

class KalapaFeatureExtraction:
    """
    Feature extraction for kalapa dataset
    """
    def __init__(self):
        logger.info('KalapaFeatureExtraction...')
        
    def fit_transform(self, train, test):
        return self.fit(train, test).transform(train, test)
    
    def fit(self, train, test, *args, **kwargs):
        logger.info('KalapaFeatureExtraction, fit')
        return self
    
    def transform(self, train, test, *args, **kwargs):
        logger.info('KalapaFeatureExtraction, transform')
        # filter variable via missing rate
        dt_s = sc.var_filter(train, y='label')
        bins = sc.woebin(dt_s, y='label', bin_num_limit=20, positive="label|1", method='tree')
        train_woe = sc.woebin_ply(train.drop(columns=['label']), bins)
        # train_woe['label'] = train['label']
        test_woe = sc.woebin_ply(test, bins)

        temp = train_woe.sample(test_woe.shape[0])
        test.reset_index(drop=True, inplace=True)
        temp.reset_index(drop=True, inplace=True)
        test_woe.fillna(temp, inplace=True)
        
        # encoding_params =  {'alpha':5, 'folds':3, 'target':train['label']}
        # train_mean, test_mean = mean_encoding(train[['8', '9', '10', '13', '35', '41', '42', '44']],
        #                      test[['8', '9', '10', '13', '35', '41', '42', '44']], 
        #                      **encoding_params)
        train = pd.concat([train, train_woe], axis=1)
        test = pd.concat([test, test_woe], axis=1)
        
        del dt_s, bins, train_woe, test_woe, temp
        gc.collect()
        return (train, test) 


        

class Encoder(object):
    @staticmethod  
    def KFoldTargetEncoding(train_data, test_data, target_name, feature_name, n_folds = 5):
        """K-FoldsTargetEncoding
        We use KFoldTargetEncoding to encoding categorical column of feature_name by target column
        We specify every values of feature column and tranform values to corresponding target encoding values

        Parameters
        ----------
        train_data: Dataframe
            Train Dataframe need to transform

        test_data: Dataframe
            Test Dataframe need to transform

        target_name: string
            Name of target column

        feature_name: string
            Name of feature column

        n_folds: int
            Number of folds

        Return
        ----------

        """
        print(type(train_data))
        # Validate variables domain
        assert (type(train_data) == pd.DataFrame), "train_data must be dataframe"
        assert (type(test_data) == pd.DataFrame), "test_data must be dataframe"
        assert (target_name != None) , "target_anme must not be null"
        assert (type(feature_name) == str),"featue_name must be string"

        train_data_cp = train_data.copy()
        test_data_cp = test_data.copy()

        # Create a kfold with n_folds
        kf = KFold(n_splits = n_folds, shuffle = True, random_state=10000)
        # Filter needed data to tranform
        # Create mapper with mean of each specific data value 
        mapper = train_data_cp.groupby(by=feature_name,axis=0)[target_name].mean()
        new_feature = train_data_cp[feature_name].copy()

        # Iterate through every fold to encoding
        for based_indx,transformed_indx in kf.split(train_data_cp):

            # Get tranformed_series and based_dataframe
            transformed_se = train_data_cp.iloc[transformed_indx][feature_name]

            based_df = pd.DataFrame({feature_name: train_data_cp.iloc[based_indx][feature_name],target_name: train_data_cp.iloc[based_indx][target_name]})

            # Get based mean of each value group
            based_means = based_df.groupby(by=feature_name, axis=0)[target_name].mean()
            # Create mapper for transforming values
            for key in based_means.index.values.tolist():
                mapper[key] = based_means[key]

            # Transform transformed_set according to value
            transformed_se = transformed_se.map(mapper)

            # Replace data 
            new_feature.update(transformed_se)


        # Create new feature in temp train data
        train_data_cp[feature_name+'_te'] = pd.to_numeric(new_feature)

        # Get different value 
        diff_vals = set(test_data[feature_name].unique()) - set(train_data[feature_name].unique())

        # Create test mapper
        test_mapper = train_data_cp.groupby(by=feature_name,axis=0)[feature_name+'_te'].mean()

        for diff_val in diff_vals:
            test_mapper[diff_val] = train_data_cp[feature_name+'_te'].mean()

        # Transform test data
        test_data_cp[feature_name] = test_data_cp[feature_name].map(test_mapper)

        # Update
        train_data_cp[feature_name].update(new_feature)
        train_data_cp=train_data_cp.drop(columns=[feature_name+'_te'])

        return train_data_cp, test_data_cp

class BPCA(base.BaseEstimator):
    """
    This class implement Bayesian Pricipal Component Analysis 

    References:
    https://pdfs.semanticscholar.org/d23c/2fc2c6fa02c1749827bb3af17cbfb3bfa4e4.pdf
    https://pdfs.semanticscholar.org/a1fb/a67f147b16e3c4bffdab3cc6f17520c74547.pdf
    https://github.com/Duuuuuu/Probabilistic-and-Bayesian-PCA/blob/master/AM_205_Final_Report.pdf
    https://github.com/Duuuuuu/Probabilistic-and-Bayesian-PCA
    """
    def __init__(self, a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3):
        # Define hyperparameters
        
        # alpha's prior (Gamma distribution)
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha

        # tau's prior (Gamma Distribution)
        self.a_tau = a_tau
        self.b_tau = b_tau

        self.beta = beta

        self.X = None
        self.pca = PCA()

    def iterations(self, N=None, batch_size=None):
        """
        Calculate iterations with respect to batch_size

        Parameters
        ----------

        N: int (required)
            The number of samples
        batch_size: int (required)
            Number of samples in each batch
        
        Return
        ----------
        iterations: int
        """
        # Validate variables
        assert (type(N) == int), 'N must be int'
        assert (type(batch_size) == int), 'batch_size must be int'

        resi = N%batch_size
        if resi == 0:
            return int(N/batch_size)
        else:
            return int(N/batch_size)+1
    
    def batch_idx(self, iter):
        if self.batch_size == self.N:
            return np.arange(self.N)
        idx1 = (iter*self.batch_size)%self.N
        idx2 = ((iter+1)*self.batch_size)%self.N
        if idx2 < idx1:
            idx2 = self.N
        return np.arange(idx1,idx2)

    def calculate_log_likelihood(self):
        """
        Calculate the log likelihood of observing self.X_b
        """
        w = self.mean_w
        c = np.eye(self.d) * self.tau + np.dot(w, w.T)
        # Normalizing X to the center point
        xc = self.X - self.X.mean(axis = 1).reshape(-1,1)
        # Covariance S of X
        s = np.dot(xc, xc.T) / self.N
        self.s = s
        c_inv_s = scipy.linalg.lstsq(c ,s)[0]
        loglikelihood = -0.5 * self.N * (self.d * np.log(2 * np.pi) + np.log(np.linalg.det(c)) + np.trace(c_inv_s))
        return loglikelihood


    def fit(self, X=None, batch_size=128, verbose=False, print_every = 5, no_repeat=True):
        """
        fit the Bayesian PCA model

        Parameters
        ----------

        X: ndarray (required)
            Dataset Nxd (samples x features) contains only numerical fields
        batch_size: int (required, default = 128)
            Number of samples in each batch. batch_size must be <= N
        verbose: bool (options, default = False)
            Print summary some information of fitting operation
        print_every: int (options, default = 5, active when verbose = True) 
            Print summary information in every print_every
            For example batch_size = 100, N =1000 then iterations =10
            Now if print_every = 2 then summary information will be print in {0,2,4,6,8,10} iterations round
        no_repeat: bool (options, default = True)
            Check if non running repeatly, if False then you want to keep previous variational parameters and missing-value place holders
        """

        # Validate variables
        assert (type(X) == np.ndarray), 'X must be Nxd numpy ndarray'
        assert (type(batch_size) == int), 'batch_size must be int'
        assert (type(verbose) == bool), 'verbose must be bool'

        # Get data X (dxN), number of features d, number of samples N, reduced dimensions q, 
        self.d = X.T.shape[0]
        self.N = X.T.shape[1]
        self.q = self.d - 1
        self.ed = []
        self.batch_size = int(min(batch_size,self.N))

        
        if (no_repeat == True) or (self.X is None):        
            # sample mean (dx1)
            self.mu = np.nanmean(X, axis=0).reshape(self.d, 1)
            # sample covariance matrix (dxd)
            self.S = np.cov(X.T).reshape(self.d, self.d)
            # SVD of S
            eigen_vas, eigen_vecs = np.linalg.eig(self.S)
            eigen_vecs = eigen_vecs[:, eigen_vas.argsort()[::-1]]
            eigen_vas = np.sort(eigen_vas)[::-1]
            # diag of sigma_square error 
            sigma_square = np.sum(eigen_vas[self.q:])/(self.d-self.q)
            # get q-dimensional of diag eigen_vas and eigen_vecs 
            eigen_vas = np.diag(eigen_vas[:self.q])
            eigen_vecs = eigen_vecs[:, :self.q]
            # Latent variables z
            self.z = self.pca.fit_transform(X[:self.batch_size, :self.q]).reshape(self.batch_size, self.q)

            # Variational parameters
            # self.mean_z = np.random.randn(self.q, self.batch_size) # latent variable
            # self.cov_z = np.eye(self.q)
            self.mean_z = self.z # latent variable
            self.cov_z = np.cov(self.z.T).reshape(self.q, self.q)
            
            # self.mean_mu = np.random.rand(self.d,1)
            # self.cov_mu = np.eye(self.d)
            self.mean_mu = self.mu
            self.cov_mu = self.S.reshape(self.d, self.d)
            
            # self.mean_w = np.random.randn(self.d, self.q)
            # self.cov_w = np.eye(self.q)
            self.mean_w = np.dot(eigen_vecs, np.sqrt(eigen_vas - sigma_square*np.eye(self.q)).reshape(self.q, self.q)).reshape(self.d, self.q)
            self.cov_w = np.cov(self.mean_w.T).reshape(self.q, self.q)

            self.a_alpha_tilde = self.a_alpha + self.d
            self.b_alpha_tilde = np.abs(np.random.randn(self.q))
            self.a_tau_tilde = self.a_tau + self.N * self.d / 2
            self.b_tau_tilde = np.abs(np.random.randn(1))
        self.X = X.T

        order = np.arange(self.N)
        iters = self.iterations(self.N, self.batch_size)
        loglikelihoods = np.zeros(iters)
        # Iterate though iters
        for it in range(iters):
            idx = order[self.batch_idx(it)]
            self.X_b = self.X[:,idx]
            self.update()
            loglikelihoods[it] = self.calculate_log_likelihood()
            if verbose and it%print_every == 0:
                print(f'Iter {it}, LL: {loglikelihoods[it]}')

        self.captured_dims()
    

    def update(self):
        """
        Update Bayesian PCA

        we calculate in X_b observations
        """
        # inverse of the sigma^2
        self.tau = self.a_tau_tilde / self.b_tau_tilde
        # hyperparameters controlling the magnitudes of each column of the weight matrix 
        self.alpha = self.a_alpha_tilde / self.b_alpha_tilde
        
        # covariance matrix of the latent variables
        self.cov_z = np.linalg.inv(np.eye(self.q) +
                    self.tau * (np.trace(self.cov_w) + np.dot(self.mean_w.T,self.mean_w)))
        # mean matrix of the latent variables
        self.mean_z = self.tau * np.dot(np.dot(self.cov_z,self.mean_w.T) , (self.X_b - self.mean_mu))

        # covariance matrix of the mean observations
        self.cov_mu = np.eye(self.d) / (self.beta + self.batch_size * self.tau)
        # mean vecotr of the mean observations
        self.mean_mu = self.tau * np.dot(self.cov_mu , np.sum(self.X_b - np.dot(self.mean_w, self.mean_z), axis =1)).reshape(self.d,1)

        # covariance matrix of each column of the weight matrix
        self.cov_w = np.linalg.inv(np.diag(self.alpha) + self.tau * 
        (self.batch_size * self.cov_z + np.dot(self.mean_z, self.mean_z.T)))
        # mean of each column of the weight matrix
        self.mean_w = self.tau * np.dot(self.cov_w, np.dot(self.mean_z, (self.X_b - self.mean_mu).T)).T

        # estimation of the b in alpha's Gamma distriution
        self.b_alpha_tilde = self.b_alpha + 0.5 * (np.trace(self.cov_w) +
                        np.diag(np.dot(self.mean_w.T, self.mean_w)))
        # estimation of the b in tau's Gamma distribution
        self.b_tau_tilde = self.b_tau + 0.5 * np.trace(np.dot(self.X_b.T, self.X_b)) + \
                0.5 * self.batch_size*(np.trace(self.cov_mu)+np.dot(self.mean_mu.flatten(),self.mean_mu.flatten())) + \
                0.5 * np.trace(np.dot(np.trace(self.cov_w)+np.dot(self.mean_w.T, self.mean_w), self.batch_size*self.cov_z+np.dot(self.mean_z, self.mean_z.T))) + \
                np.sum(np.dot(np.dot(self.mean_mu.flatten(), self.mean_w), self.mean_z)) + \
                -np.trace(np.dot(self.X_b.T, np.dot(self.mean_w, self.mean_z))) + \
                -np.sum(np.dot(self.X_b.T, self.mean_mu))
        

    def transform(self, X=None, full=True):
        """
        Transform observation samples from the fitted model to latent variables

        Parameters
        ----------

        X: ndarray (required)
            Dataset Nxd (samples x features) contains only numerical fields
        full: bool (options, default = True)
            If true the using q = d -1 dimensional principal components
            If false the using self.ed to controll the dimentional principal components needed
        """
        X = self.X if X is None else X.T
        
        if full:
            w = self.mean_w
            q = self.q
        else:
            w = self.mean_w[:,self.ed]
            q = len(self.ed)

        m = np.eye(q) * self.tau + np.dot(w.T, w)
        inv_m = np.linalg.inv(m)
        z = np.dot(np.dot(inv_m, w.T), X - self.mean_mu)
        return z.T
    
    def inverse_transform(self, z, full=True):
        """
        Transform the latent variables into observation samples

        Parameters
        ----------

        z: ndarray (required)
            Latent dataset Nxq (samples x features) contrains only numerical fields
        full: bool (options, default = True)
            If true the using q = d -1 dimensional principal components
            If false the using self.ed to controll the dimentional principal components needed
        """
        z = z.T
        
        if full:
            w = self.mean_w
        else:
            w = self.mean_w[:,self.ed]

        x = np.dot(w, z) + self.mean_mu
        return x.T

    def captured_dims(self):
        """
        The number of captured dimensions
        
        """
        sum_alpha = np.sum(1/self.alpha)
        self.ed = np.array([i for i, inv_alpha in enumerate(1/self.alpha) if inv_alpha < sum_alpha/self.q])
        
class Imputer(object):
    def __init__(self):
        self._pca = BPCA()
    
    def fit(self, data=None,batch_size=100, epochs = 10, early_stopping = 20, err_threshold = 1e-5, full_dimens = True, verbose=False, print_every=10):
        """
        Fit observations 

        Parameters
        ----------

        data: ndarray (required)
            Dataset Nxd (samples x features) contains only numerical fields
        batch_size: int (required, default = 100)
            Number of samples in each batch. batch_size must be <= N
        epochs: int (required, default = 100)
            The number of times running algorithms
        early_stopping: int
            The number of times that will break algorithm if there is no more change in BPCA 
        err_threshold: float (required, default = 1e-2)
            Threshold used for stopping running if error residual (mse) < err_threshold
        verbose: bool (options, default = False)
            Print summary some information of fitting operation
        print_every: int (options, active when verbose = True) 
            Print summary information in every print_every
            For example batch_size = 100, N =1000 then iterations =10
            Now if print_every = 2 then summary information will be print in {0,2,4,6,8,10} iterations round
        full_dimens: bool (options, default = True)
            If true the using q = d -1 dimensional principal components
            If false the using self.ed to controll the dimentional principal components needed
        """

        # Validate variable
        assert (type(data) == np.ndarray),'data must be ndarray Nxd (samples x features)'
        assert (type(batch_size) == int),'batch_size must be int'
        assert (type(epochs) == int),'epochs must be int'
        assert (type(full_dimens) == bool),'full_dimens must be bool'
        assert (type(verbose) == bool),'verbose must be bool'
        assert (type(print_every) == int),'print_every must be int'
        

        # Data (Nxd)
        _data = data.copy()
        # Missing (Nxd) {False, True}
        _missing = np.isnan(_data)
        # Obsered (Nxd) {False, True}
        _observed = ~_missing
        _prev_mse = np.inf

        row_defau = np.zeros(_data.shape[1])
        row_means = np.repeat(np.nanmean(_data, axis=0, out=row_defau).reshape(1,-1),_data.shape[0], axis=0)
        
        _data[_missing] = row_means[_missing]
        _data = np.nan_to_num(_data)

        early_stopping_count = 0

        for epoch in range(epochs):

            self._pca.fit(X=_data,batch_size=batch_size,verbose=verbose, print_every = print_every, no_repeat=False)

            temp = self._pca.inverse_transform(self._pca.transform(_data, full=full_dimens), full=full_dimens)

            mse = np.sum((_data[_observed] - temp[_observed])**2)/_data.shape[0]
            
            mse_residual = mse - _prev_mse
            if np.abs(mse_residual) < err_threshold:
                early_stopping_count +=1
                if (early_stopping_count >= early_stopping):
                    break
            else:
                early_stopping_count = 0
            
            if mse_residual < 0:
                _data[_missing] = temp[_missing]
                _prev_mse = mse

            if verbose:
                print(f'Epoch {epoch} Mean squared estimation: {_prev_mse}')            
        
        gc.enable()
        del _missing, _observed, _data, _prev_mse, temp
        gc.collect()

        return self


    def transform(self, data=None, full_dimens=True):
        """
        Tranforms missing data

        Parameters
        ----------
        data: ndarray (required)
            Dataset Nxd (samples x features) contains only numerical fields
        """
        # Validate variables
        assert (type(data) == np.ndarray), 'data must be numpy ndarray'
        
        # Data (Nxd)
        _data = data.copy()
        # Missing (Nxd) {False, True}
        _missing = np.isnan(data)
        # Obsered (Nxd) {False, True}
        _observed = _missing
        
        row_defau = np.zeros(_data.shape[1])
        row_means = np.repeat(np.nanmean(_data, axis=0, out=row_defau).reshape(1,-1),_data.shape[0], axis=0)
        
        _data[_missing] = row_means[_missing]
        _data = np.nan_to_num(_data)

        _data[_missing] = self._pca.inverse_transform(self._pca.transform(_data, full=full_dimens), full=full_dimens)[_missing]

        gc.enable()
        del _missing, _observed
        gc.collect()

        return _data


    def fit_transform(self, data=None,batch_size=100, epochs = 10, full_dimens = True, verbose=False, print_every=10):
        """
        Fit observations and transform missing data

        Parameters
        ----------

        data: ndarray (required)
            Dataset Nxd (samples x features) contains only numerical fields
        batch_size: int (required, default = 100)
            Number of samples in each batch. batch_size must be <= N
        epochs: int (required, default = 100)
            The number of times running algorithms
        verbose: bool (options, default = False)
            Print summary some information of fitting operation
        print_every: int (options, active when verbose = True) 
            Print summary information in every print_every
            For example batch_size = 100, N =1000 then iterations =10
            Now if print_every = 2 then summary information will be print in {0,2,4,6,8,10} iterations round
        full_dimens: bool (options, default = True)
            If true the using q = d -1 dimensional principal components
            If false the using self.ed to controll the dimentional principal components needed
        """
        return self.fit(data=data, batch_size=batch_size, epochs=epochs, full_dimens=full_dimens, verbose=verbose, print_every=print_every).transform(data)

class GlobalClosestFit:
  """
  Using global closest fit method for replacing missing values
  """
  def fit(self, X, *args, **kwargs):
    """
    Using Blocks of Attribute-Value Pairs and Characteristic Sets to find sets of cases that are not distinguish for each case
    then calculate the closest observation using distance fomula in _distance function

    Parameters
    -----------
    X: DataFrame [n samples, d features] (required)
      The input data with missing values
    """
    print('GlobalClosestFit, fit')
    
    print('GlobalClosestFit, Blocks of Attribute-Value, define np.nan as "do not care"')
    blocks_ = dict()
    for fea in X:
      idx_null = set(X[X[fea].isnull()].index)
      for value in X[fea].unique():
        if value is not np.nan:
          blocks_[(fea, value)] = set(X[X[fea] == value].index).union(idx_null)
    print('Characteristic set')
    K_ = dict()
    for fea in X:
      for value in X[fea].unique():
        if not pd.isna(value):
          indices = X[X[fea] == value].index
          for idx in indices:
            if idx not in K_.keys():
              K_[idx] = blocks_[(fea, value)]
            else:
              K_[idx] = K_[idx].intersection(blocks_[(fea, value)])    
    # for idx in range(X.shape[0]):
    #   for fea in X:
    #     if not pd.isna(X[fea].iloc[idx]):
    #       indices = blocks_.get((fea, X[fea].iloc[idx]))
    #       if idx not in K_.keys():
    #         K_[idx] = indices
    #       else:
    #         K_[idx] = K_[idx].intersection(indices)
    del blocks_
    gc.collect()
    print('GlobalClosestFit, Calculate the closest fit')    
    r = [(np.nanmax(X[fea]) - np.nanmin(X[fea])) if (X[fea].dtype.name == 'float32') else np.nan for fea in X]
    
    self.closest_dict_ = {}
    pseudo_max = np.nanmax(r)*9999

    for idx, row in X.iterrows():
      print(idx)
      min_dist = pseudo_max
      curr_idx = -1
      
      closest_indices = K_[idx]
      for idx1 in closest_indices:
        if(idx != idx1):
          row1 = X.iloc[idx, :]
          dist = self._distance(row, row1, r)
          if dist < min_dist:
            min_dist = dist
            curr_idx = idx1

      self.closest_dict_[int(idx)] = int(curr_idx)
    print('GlobalClosestFit, done fit')
    del K_, r
    gc.collect()
    return self  

  def transform(self, X, *args, **kwargs):
    """
    Filling missing values with its closest observation, there would be some fields still missing

    Parameters
    ----------
    X: DataFrame [n samples, d features] (required)
      The input data with missing values, this input data must be the same data in calling fit function
    """
    print('GlobalClosestFit, transform')
    X_cp = X.copy()
    for idx, _ in X_cp.iterrows():
      if int(self.closest_dict_[idx]) != -1:
        closest_ = X_cp.iloc[int(self.closest_dict_[idx]), :]
        for idx_obs in range(len(X_cp.iloc[idx, :])):
          if pd.isna(X_cp.iloc[idx, idx_obs]):
            X_cp.iloc[idx, idx_obs] = closest_[idx_obs]
    
    print('GlobalClosestFit, done transform')
    return X_cp
  
  def fit_transform(self, X, *args, **kwargs):
    return self.fit(X).transform(X)
  
  def _distance(self, x, y, r):
    """
    Calculate distance(x, y) (x,y are vectors) with following formula:
    distance(x, y) = sum(distance(xi, yi)), where xi, yi are element of vector x and vector y

    distance(xi, yi):
      0: if xi = yi
      1: if x and y are symbolic and xi != yi, or xi = nan or yi = nan
      |xi-yi| / r: if xi and yi are numeric and xi != yi, where r is the interval between maximmum and minimum of known values of current feature
    
    """
    
    distance = 0
    for idx in range(len(x)):
      if pd.isna(x.iloc[idx]) or pd.isna(y.iloc[idx]):
        distance += 1
      elif (type(x.iloc[idx]) == str) and (x.iloc[idx] != y.iloc[idx]):
        distance += 1
      elif (type(x.iloc[idx]) == float) and (x.iloc[idx] != y.iloc[idx]):
        distance += np.abs(x.iloc[idx] - y.iloc[idx]) / r[idx]
    
    return distance