from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

class CSPlot(object):
    @staticmethod
    def plot_pre_rec_curve(y_true, y_pred):
        precision, recall, thres = precision_recall_curve(y_true, y_pred)
        plt.figure(figsize=(10,8))
        plt.plot(thres, precision[:-1], 'b--', label='precision')        
        plt.plot(thres, recall[:-1], 'g-', label='recall')
        plt.xlabel('Threshold')
        plt.ylabel('Probability')
        plt.title('Precision and Recall') 
        plt.legend()