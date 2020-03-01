import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import gc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve, make_scorer, accuracy_score, roc_curve, roc_auc_score

class cs_metrics:
    @staticmethod
    def cross_validate_auc(model, X, y, kfold, *fit_args):
      validate_aucs = []
      for train_index, validate_index in kfold.split(X):
        # get train and test split
        if isinstance(X, pd.DataFrame):
          X_train, X_validate = X.iloc[train_index], X.iloc[validate_index]
          y_train, y_validate = y[train_index], y[validate_index]
        elif isinstance(X, np.ndarray):
          X_train, X_validate = X[train_index], X[validate_index]
          y_train, y_validate = y[train_index], y[validate_index]
        # train model
        model.fit(X_train, y_train, *fit_args)
        # predit proba
        y_validate_scores = model.predict_proba(X_validate)[:,-1]
        # calculate AUC
        validate_auc = roc_auc_score(y_validate, y_validate_scores)
        # append AUC to AUC lists
        validate_aucs.append(validate_auc)

      validate_aucs = np.array(validate_aucs)
      gini_results = validate_aucs * 2 - 1
      print("------- Cross Validation AUC---------")
      print(validate_aucs)
      print("------------- Mean AUC --------------")
      print(np.mean(validate_aucs))
      print()
      print("------- Cross Validation Gini---------")
      print(gini_results)
      print("------------- Mean Gini --------------")
      print(np.mean(gini_results))
    @staticmethod
    def max_f1_score(y_true, y_proba):
      precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
      f1_scores = 2*recall*precision/(recall+precision)
      return np.nanmax(f1_scores)

    @staticmethod
    def cross_validate_brier(model, X, y):
      cv_results = cross_val_score(model, X, y, scoring="neg_brier_score", cv=5, n_jobs=-1)
      print("------- Cross Validation Brier---------")
      print(cv_results)
      print("------------- Mean Brier --------------")
      print(np.mean(cv_results))


class cs_plots:
    # get predictions from model
    # return probabilities
    @staticmethod
    def get_predictions_from_model(model, x_validate):
        y_probabilities = model.predict_proba(x_validate)
        y_labels = np.argmax(y_probabilities, axis=1)
        return y_probabilities[:,1], y_labels

    # plot auc all
    @staticmethod
    def plot_auc_all(model, x_train, y_train, x_validate, y_validate):
        # calculate probabilities
        y_train_scores, y_train_labels = cs_plots.get_predictions_from_model(model, x_train)
        y_validate_scores, y_validate_labels = cs_plots.get_predictions_from_model(model, x_validate)

        # calculate false positive rate & true positive rate
        fpr_train, tpr_train, thresh_train = roc_curve(y_train, y_train_scores)
        fpr_validate, tpr_validate, thresh_validate = roc_curve(y_validate, y_validate_scores)

        # print average precision & auc scores
        print("Train set")
        print("Accuracy Score {}".format(accuracy_score(y_train,y_train_labels)))
        auc_train = roc_auc_score(y_train, y_train_scores)
        print("AUC Score {}".format(auc_train))
        print("Gini Score {}".format(2*auc_train-1))
        print()
        print("Validation set")
        auc_val = roc_auc_score(y_validate, y_validate_scores)
        print("Accuracy Score {}".format(accuracy_score(y_validate,y_validate_labels)))
        print("AUC Score {}".format(auc_val))
        print("Gini Score {}".format(2*auc_val-1))

        plt.figure(figsize=(12,10))
        plt.plot(fpr_train, tpr_train, label="train set")
        plt.plot(fpr_validate, tpr_validate, label="validation set")
        plt.plot([0,1],[0,1], "k--", label="naive prediction")
        plt.axis([0,1,0,1])
        plt.legend(loc="best")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive rate")
    # Plot KFold auc
    @staticmethod
    def plot_kfold_auc(model, X, y, kfold, *fit_args):

      for train_index, validate_index in kfold.split(X):
        # get train and test split
        if isinstance(X, pd.DataFrame):
          X_train, X_validate = X.iloc[train_index], X.iloc[validate_index]
          y_train, y_validate = y[train_index], y[validate_index]
        elif isinstance(X, np.ndarray):
          X_train, X_validate = X[train_index], X[validate_index]
          y_train, y_validate = y[train_index], y[validate_index]

        cs_plots.plot_auc_all(model, X_train, y_train, X_validate, y_validate)
        print('\n\n')
    @staticmethod
    def corr_plot(df):
        """
        Plotting correlation figure

        Parameters
        ----------
        df: DataFrame (required)
          The dataframe Nxd (samples x features) used to plotting correlation

        """
        # Validate variables
        assert (type(df) == pd.DataFrame), 'df must be pandas DataFrame'

        # Setting configurations
        height = df.shape[1]
        plt.figure(figsize=(height+10,height+10))

        # Calculate correlations
        corr = df.corr()

        # Plotting
        sns.heatmap(corr, cmap=plt.cm.RdYlBu_r, vmin=-.25, annot=True, vmax=0.6)
        plt.title('Correlation matrix')

        gc.enable()
        del corr
        gc.collect()

    @staticmethod
    def plot_feature_importances(importances, labels):
        """
        Plot importance features returned by a model. This can work with any measure of
        feature importance provided that higher importances is better.

        Parameter
        --------
        importances: ndarray (required, size N)
          the numpy data array of importance values
        lables: list of string (required, size N)
          the list of names corresponding to each of importances
        """
        # Validate variable
        assert (len(importances) == len(labels)), 'size of importances must be equal to size of lables'

        # Create impotances dictionary
        importances_dic = {}

        for i in range(len(importances)):
          importances_dic[importances[i]] = labels[i]
        sort_values = sorted(importances_dic, reverse=True)

        sort_dic = {}
        for value in sort_values:
          sort_dic[value] = importances_dic[value]

        # Make a horizontal bar chart of feature importances
        plt.figure(figsize=(10,len(importances)))

        sns.barplot(list(sort_dic.keys()),list(sort_dic.values()))

    @staticmethod
    def Visualization(data, target):
        """
        Visualize data base on the following dtypes scheme
          - category (category, object): histogram
          - numerical (float64, int64, float32, int32): scatter
          - boolean (boolean): histogram

        Parameters
        ----------
        data: DataFrame (required)
          The data set is used to visualization
        target: str (required)
          Name of target column
        """
        # Validate variables
        assert(type(data) == pd.DataFrame),'data must be pandas DataFrame'
        assert(type(target) == str), 'target must be str'

        # Create dictionary of dataframe corresponding to each dtype
        select_data = {}
        select_data['category'] = list(data.select_dtypes(include=['category','object']).columns)
        select_data['numerical'] = list(data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns)
        select_data['boolean'] = list(data.select_dtypes(include=['bool']).columns)

        # Calculate maximmum columns
        max_rows = 0
        for df in select_data.keys():
          max_rows = max(max_rows, len(select_data[df]))

        # Create multiple figure
        fig, axs = plt.subplots(nrows=max_rows, ncols=3,figsize=(50,350))

        # Sort data base on TARGET ascending order
        data_vi = data.sort_values(by=target)

        # Visualize category
        for ax_col, col_na in enumerate(select_data['category']):
          a = sns.countplot(x=col_na, hue='label', data=data_vi, ax=axs[ax_col, 0])
          for p in a.patches:
            value = '{}'.format(p.get_height())
            x = p.get_x() + p.get_width()/2
            y = p.get_y() + p.get_height() +0.02
            a.annotate(value, (x, y))

        # Visualize numerical
        for ax_col, col_na in enumerate(select_data['numerical']):
          sns.scatterplot(x=data_vi.index, y=col_na, hue='label', style='label', palette="Set2", data=data_vi, ax=axs[ax_col, 1])

        # Visulize boolean
        for ax_col, col_na in enumerate(select_data['boolean']):
          a = sns.countplot(x=col_na, hue='label', data=data_vi, ax=axs[ax_col, 2])
          for p in a.patches:
            value = '{}'.format(p.get_height())
            x = p.get_x() + p.get_width()/2
            y = p.get_y() + p.get_height() +0.02
            a.annotate(value, (x, y))

        gc.enable()
        del data_vi
        gc.collect()
