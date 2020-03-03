import numpy as np
import gc
from sklearn import base
import scipy

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
            # Variational parameters
            self.mean_z = np.random.randn(self.q, self.batch_size) # latent variable
            self.cov_z = np.eye(self.q)
            self.mean_mu = np.random.rand(self.d,1)
            self.cov_mu = np.eye(self.d)
            self.mean_w = np.random.randn(self.d, self.q)
            self.cov_w = np.eye(self.q)
            self.a_alpha_tilde = self.a_alpha + self.d
            self.b_alpha_tilde = np.abs(np.random.randn(self.q))
            self.a_tau_tilde = self.a_tau + self.N * self.d / 2
            self.b_tau_title = np.abs(np.random.randn(1))
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
                print(f'Iter {it}, LL: {loglikelihoods[it]}, alpha: {str(self.alpha)}')

        self.captured_dims()
    

    def update(self):
        """
        Update Bayesian PCA

        we calculate in X_b observations
        """
        # inverse of the sigma^2
        self.tau = self.a_tau_tilde / self.b_tau_title
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
    
    def fit(self, data=None,batch_size=100, epochs = 10, full_dimens = True, verbose=False, print_every=10):
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
        _mse = [np.inf]

        row_defau = np.zeros(_data.shape[1])
        row_means = np.repeat(np.nanmean(_data, axis=0, out=row_defau).reshape(1,-1),_data.shape[0], axis=0)
        
        _data[_missing] = row_means[_missing]
        _data = np.nan_to_num(_data)

        for epoch in range(epochs):

            self._pca.fit(X=_data,batch_size=batch_size,verbose=verbose, print_every = print_every, no_repeat=False)

            temp = self._pca.inverse_transform(self._pca.transform(_data, full=full_dimens), full=full_dimens)

            mse = np.sum((_data[_observed] - temp[_observed])**2)/_data.shape[0]
            
            if np.abs(mse - _mse[-1]) < 1e-3:
                break

            if mse < _mse[-1]:
                _data[_missing] = temp[_missing]
                _mse.append(mse)

            if verbose:
                print(f'Epoch {epoch} Mean squared estimation: {_mse[-1]}')            
        
        gc.enable()
        del _missing, _observed, _data
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
