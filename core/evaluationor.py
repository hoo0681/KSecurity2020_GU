"""
you can evaulate the predict result If you have a label(answer) file.
Compare predict file and label(answer) file.

Thresdhold(args.threshold) change the predict score.
Default value of threshold is 0.7

Save options is not completed. I will save the data to confusion_matrix picture.
"""
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
import pandas as pd
import numpy as np
import argparse
import tqdm
import os
import sys
import matplotlib.pyplot as plt
import logging
import itertools
import matplotlib
matplotlib.use('QT5Agg')

#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_qt5agg import FigureCanvas 
#from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

logger = logging.getLogger('Evaluating')
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-5s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Evaluator:
    def __init__(self, testcsv, label):
        logger.debug("evaluator constructor Start")
        self.testdata = pd.read_csv(testcsv, names=['hash', 'y_pred']).sort_values(by=['hash'])
        self.labeldata = pd.read_csv(label, names=['hash', 'y']).sort_values(by=['hash'])
        logger.debug('testdatacsv: {}'.format(testcsv))
        logger.debug('testdatalabel: {}'.format(label))
        logger.debug("evaluator constructor is Done")

    def plot(self, cm):
        """
        http://www.tarekatwan.com/index.php/2017/12/how-to-plot-a-confusion-matrix-in-python/
        """
        plt.clf()
        #plt.ion()
        #plt.close(1)
        #plt.close(2)

        plt.figure(1)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
        classNames = ['Malware','Beign']
        plt.title('Malware Detection')
        plt.ylabel('Test')
        plt.xlabel('Label')
        plt.colorbar()
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)
        s = [['TP','FP'], ['FN', 'TN']]
        
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.figure(2)
        plt.plot(self.fpr,self.tpr,'o-',ms=2,label="Logistic Regression")
        plt.plot([0, 1], [0, 1], 'k--', label="random guess")
        #ax2.plot([fallout], [recall], 'ro', ms=10)
        plt.xlabel('(Fall-Out)')
        plt.ylabel('(Recall)')
        plt.title('Receiver operating characteristic example')
        sys.exit(plt.show())
        
    def run(self):
        y = []
        ypred = self.testdata.y_pred
        end = len(self.testdata)

        if len(self.labeldata) == end:
            try:
                for idx, row in tqdm.tqdm(self.testdata.iterrows(),ascii=True, total=end):
                    _name = row['hash']
                    r = self.labeldata[self.labeldata.hash==_name].values[0][1]
                    y.append(r)


                # #get and print accuracy
                accuracy = accuracy_score(y, ypred)
                self.fpr, self.tpr, thresholds = roc_curve(y, ypred)
                print("accuracy : %.2f%%" % (np.round(accuracy, decimals=4)*100))
            
                #get and print matrix
                tn, fp, fn, tp = confusion_matrix(y, ypred).ravel()
                mt = np.array([[tp, fp],[fn, tn]])
                acc=float((tp+tn)/(tp+fn+fp+tn))
                Precision=float((tp)/(tp+fp))
                Recall=float((tp)/(tp+fn))
                MCC= ((tp*tn-fp*fn)/float(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5))
                F1_score=((2*tp)/float(2*tp+fp+fn))

                logger.info(mt)
                print('https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7')
                print("(MCC)Matthews correlation coefficient \n [-1 ~ +1] +1 is best!! score 0 means random classifier")
                print("(nMCC)normalized Matthews correlation coefficient \n [0~1] 1 is best!! 0.5 means random classifier")
                print("Accuracy : %.2f%%" % ( round(acc, 4) * 100))
                print('Precision and Recall are trade-off \n Precis : Recal = %.2f%% : %.2f%%' % ( (round(Precision, 4) * 100), (round(Recall, 4) * 100)))
                print("false postive rate : %.2f%%" % ( round(fp / float(fp + tn), 4) * 100))
                print("false negative rate : %.2f%%" % ( round(fn / float(fn + tp), 4) * 100))
                print("MCC : %.2f" % (MCC))
                print("nMCC: %.2f" % ((MCC+1)/2))
                print("F1 score: %.2f" % (F1_score))
                logger.info('******eval DONE******')
                # run plot
                self.plot(mt)

            except:
                logger.info("[Error] Please Check label file ****** ")
                return
        else:
            logger.info("[Error] Please Check label file ****** ")