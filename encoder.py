import pandas as pd
from sklearn.model_selection import KFold

class Encoder():
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
