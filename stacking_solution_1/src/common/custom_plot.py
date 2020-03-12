import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve 
from .custom_metrics import CSMetrics

class CSPlot:

    # plot auc all
    @staticmethod
    def plot_auc_all( 
                    y_train_labels, 
                    y_train_scores,
                    y_dev_labels,
                    y_dev_scores):

        # calculate false positive rate & true positive rate
        fpr_train, tpr_train, thresh_train = roc_curve(y_train_labels, y_train_scores)
        fpr_validate, tpr_validate, thresh_validate = roc_curve(y_dev_labels, y_dev_scores)

        CSMetrics.auc_train_dev(y_train_labels, y_train_scores, y_dev_labels, y_dev_scores)
        
        plt.figure(figsize=(12,10))
        plt.plot(fpr_train, tpr_train, label="train set")
        plt.plot(fpr_validate, tpr_validate, label="validation set")
        plt.plot([0,1],[0,1], "k--", label="naive prediction")
        plt.axis([0,1,0,1])
        plt.legend(loc="best")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive rate")
 