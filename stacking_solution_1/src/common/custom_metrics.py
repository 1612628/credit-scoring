import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import gc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve, make_scorer, accuracy_score, roc_curve, roc_auc_score

class CSMetrics(object):
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

    @staticmethod
    def auc_train_dev(y_train_labels,
                      y_train_preds,
                      y_dev_labels,
                      y_dev_preds):
    
      print("Train set")
      auc_train = roc_auc_score(y_train_labels, y_train_preds)
      print("AUC Score {}".format(auc_train))
      print("Gini Score {}".format(2*auc_train-1))
      print()
      print("Dev set")
      auc_val = roc_auc_score(y_dev_labels, y_dev_preds)
      print("AUC Score {}".format(auc_val))
      print("Gini Score {}".format(2*auc_val-1))


