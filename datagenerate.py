import pandas as pd

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

