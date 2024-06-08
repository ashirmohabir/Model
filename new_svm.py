
from __future__ import print_function, division
from builtins import range

from sklearn.preprocessing import StandardScaler
from datetime import datetime

from sklearn import metrics
from sklearn.metrics import roc_curve

import numpy as np
import matplotlib.pyplot as plt
import itertools

from CICDS_pipeline import cicidspipeline
from CICDS_pipeline_poison import cicids_poisoned_pipeline

class otherLinearSVM:
  
  def __init__(self, C=1.0):
    self.C = C

  def _objective(self, margins):
    return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1 - margins).sum()

  def fit(self, X, Y, lr=1e-5, n_iters=400):
    N, D = X.shape
    self.N = N
    self.w = np.random.randn(D)
    self.b = 0

    # gradient descent
    losses = []
    for _ in range(n_iters):
      margins = Y * self._decision_function(X)
      loss = self._objective(margins)
      losses.append(loss)
      
      idx = np.where(margins < 1)[0]
      grad_w = self.w - self.C * Y[idx].dot(X[idx])
      self.w -= lr * grad_w
      grad_b = -self.C * Y[idx].sum()
      self.b -= lr * grad_b

    self.support_ = np.where((Y * self._decision_function(X)) <= 1)[0]
    print("num SVs:", len(self.support_))

    print("w:", self.w)
    print("b:", self.b)

    plt.plot(losses)
    plt.title("loss per iteration")
    plt.show()

  def _decision_function(self, X):
    return X.dot(self.w) + self.b

  def predict(self, X):
    return np.sign(self._decision_function(X))

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(Y == P)


def plot_decision_boundary(model, X, Y, resolution=100, colors=('b', 'k', 'r')):
  np.warnings.filterwarnings('ignore')
  fig, ax = plt.subplots()

  # Generate coordinate grid of shape [resolution x resolution]
  # and evaluate the model over the entire space
  x_range = np.linspace(X[:,0].min(), X[:,0].max(), resolution)
  y_range = np.linspace(X[:,1].min(), X[:,1].max(), resolution)
  grid = [[model._decision_function(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
  grid = np.array(grid).reshape(len(x_range), len(y_range))
  
  # Plot decision contours using grid and
  # make a scatter plot of training data
  ax.contour(x_range, y_range, grid.T, (-1, 0, 1), linewidths=(1, 1, 1),
             linestyles=('--', '-', '--'), colors=colors)
  ax.scatter(X[:,0], X[:,1],
             c=Y, lw=0, alpha=0.3, cmap='seismic')
  
  # Plot support vectors (non-zero alphas)
  # as circled points (linewidth > 0)
  mask = model.support_
  ax.scatter(X[:,0][mask], X[:,1][mask],
             c=Y[mask], cmap='seismic')

  # debug
  ax.scatter([0], [0], c='black', marker='x')

  plt.show()



def runCICIDS(model):

    cipl = cicidspipeline()
    poisoned_pipeline = cicids_poisoned_pipeline()
    X_train, y_train, X_test, y_test = cipl.cicids_data_binary()
    print('dataset has been split into train and test data')
    X_poisoned_train, y_poisoned_train, X_poisoned_test, y_poisoned_test = poisoned_pipeline.cicids_data_binary()
    print('dataset has been split into poisoned train and test data')


    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    y_poisoned_train[y_poisoned_train == 0] = -1
    y_poisoned_test[y_poisoned_test == 0] = -1
    scaler = StandardScaler()



    X_poisoned_train = scaler.fit_transform(X_poisoned_train)
    X_poisoned_test = scaler.transform(X_poisoned_test)

  # now we'll use our custom implementation
    model = otherLinearSVM(C=1.0)

    t0 = datetime.now()
    model.fit(X_train, y_train, lr=1e-04, n_iters= 400)
    cm = cmSklearn(model, y_test, X_test)
    trainAcc = model.score(X_train, y_train)
    print("train score:", trainAcc,  "duration:", datetime.now() - t0)
    t0 = datetime.now()
    testAcc = model.score(y_train, y_test)
    print("test score:", testAcc, "duration:", datetime.now() - t0)
    targetNames = ['Normal', 'Intrusion']
    plot_confusion_matrix(model, cm,targetNames, title = 'Confusion Matrix')
    if X_train.shape[1] == 2:
        plot_decision_boundary(model, X_train, y_train)
    print(metrics.classification_report(y_test, model.predict(y_train)))
    plotRoc(model, y_train, y_test)
    return testAcc





def cmSklearn(model, expected, X):
  P = model.predict(X) 
  P = np.nan_to_num(P)
  cm = metrics.confusion_matrix(expected, P)
  return cm

def plot_confusion_matrix(model, cm, testy, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    
        plt.figure(figsize=(5,5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tickmarks = np.arange(len(testy))
        plt.xticks(tickmarks, testy, rotation=45)
        plt.yticks(tickmarks, testy)

        if normalize:

            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            cm = np.around(cm, decimals=2)

            cm[np.isnan(cm)] = 0.0

            print("Normalized confusion matrix")

        else:

            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            plt.text(j, i, cm[i, j],

                    horizontalalignment="center",

                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel("Predicted Label")
        plt.ylabel("True label")
        plt.show()


def plotRoc(model, testx, testy):
  ns_probs = [0 for _ in range(len(testy))]
  P = model.predict(testx)
  P = np.nan_to_num(P)
  plt.title("ROC Curve for Support Vector Machine model")
        # calculate roc curves
  ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
  lr_fpr, lr_tpr, _ = roc_curve(testy, P)
        # plot the roc curve for the model
  plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
  plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
        # show the legend
  plt.legend()
        # show the plot
  plt.show()
    