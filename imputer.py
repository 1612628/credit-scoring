import numpy as np
from .bpca import BPCA
import gc

class Imputer():
    def __init__(self):
        self._pca = BPCA()
    
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

        # Validate variable
        assert (type(data) == np.ndarray),'data must be ndarray Nxd (samples x features)'
        assert (type(batch_size) == int),'batch_size must be int'
        assert (type(epochs) == int),'epochs must be int'
        assert (type(full_dimens) == bool),'full_dimens must be bool'
        assert (type(verbose) == bool),'verbose must be bool'
        assert (type(print_every) == int),'print_every must be int'
        

        # Data (Nxd)
        self._data = data.copy()
        # Missing (Nxd) {False, True}
        self._missing = np.isnan(data)
        # Obsered (Nxd) {False, True}
        self._observed = ~self._missing
        self._mse = np.zeros(epochs)

        row_defau = np.zeros(self._data.shape[1])
        row_means = np.repeat(np.nanmean(self._data, axis=0, out=row_defau).reshape(1,-1),self._data.shape[0], axis=0)
        
        self._data[self._missing] = row_means[self._missing]
        self._data = np.nan_to_num(self._data)

        for epoch in range(epochs):

            self._pca.fit(X=self._data,batch_size=batch_size,verbose=verbose, print_every = print_every)

            temp = self._pca.inverse_transform(self._pca.transform(self._data, full=full_dimens), full=full_dimens)
            self._data[self._missing] = temp[self._missing]

            self._mse[epoch] = np.sum((self._data[self._observed] - temp[self._observed])**2)/self._data.shape[0]

            if verbose:
                print(f'Epoch {epoch} Mean squared estimation: {self._mse[epoch]}')            
        
        gc.enable()
        del self._missing, self._observed
        gc.collect()

        return self._data
