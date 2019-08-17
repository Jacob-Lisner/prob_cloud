import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import math
import re
import sklearn.linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from sklearn.kernel_approximation import RBFSampler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from sklearn.preprocessing import normalize

import itertools
from scipy import linalg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from geneticSelect import geneticSelect

import prob_cloud


#from https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def main():

    map = "2015"

    clusterData = pd.read_csv("sets/reduced_set_" + map + ".csv")
    xData = pd.read_csv("sets/no_suicide_set_" + map + ".csv")
    yFData = pd.read_csv("sets/female_rate_" + map + ".csv", header = None)
    yMData = pd.read_csv("sets/male_rate_" + map + ".csv", header = None)


    xlabels = xData.values[:,0]
    yflabels = yFData.values[:,0]
    ymlabels = yMData.values[:,0]
    flabels = xData.columns.values[1:]
    #print(flabels)
    #print(xlabels)
    #print(yflabels)

    xdata = xData.values[:,1:]
    yfdata = yFData.values[:,1:]
    ymdata = yMData.values[:,1:]

    features = xdata.shape[1]
    samples = xdata.shape[0]


    xp = normalize(xdata[:,0].reshape(-1,1), axis = 0, norm = "max")
    yp = normalize(xdata[:,1].reshape(-1,1), axis = 0, norm = "max")
    #p3 = normalize(xdata[:,2].reshape(-1,1), axis = 0, norm = "max")
    zp = normalize(yfdata, axis = 0, norm = "max")

    xFeed = np.append(xp,yp, axis = 1)

    method = MLPRegressor(hidden_layer_sizes = (125,50,25), activation = "tanh", solver = "adam", learning_rate = "adaptive", random_state = 0)
    cloud = prob_cloud.cloud(method)

    xVals = xFeed
    yVals = yfdata
    cloud.fit(xVals,yVals)

    confs = cloud.predict_confidence(xVals,yVals)
    fig = plt.figure()
    ax = fig.add_subplot()
    i = 0
    for i in range(2, xdata.shape[1]):
        xp = normalize(xdata[:,i].reshape(-1,1), axis = 0, norm = "max")
        yp = confs

        xFeed = np.append(xp,yp, axis = 1)
        method = MLPRegressor(hidden_layer_sizes = (125,50,25), activation = "tanh", solver = "adam", learning_rate = "adaptive", random_state = 0)
        cloud = prob_cloud.cloud(method, pred_type = "mean")
        cloud.fit(xFeed,yfdata)
        print(mean_squared_error(cloud.predict(xFeed),yfdata))
        confsNew = cloud.predict_confidence(xFeed,yfdata)
        confs = np.maximum(confs,confsNew)
        if(i%10 == 0):
            ax.scatter(normalize(confs,axis = 0, norm = "max"),yVals.astype(float), label = i)





    #ax = fig.add_subplot(111, projection = '3d')


    ax.legend()
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Rates')
    #ax.set_zlabel('Rates')

    plt.show()

if __name__ == "__main__":
  main()
