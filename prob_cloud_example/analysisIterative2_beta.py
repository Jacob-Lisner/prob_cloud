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

    xdataBackup = xdata
    yfdataBackup = yfdata

    train_X, val_X, train_Y, val_Y = train_test_split(xdata, yfdata, test_size = 0.3, random_state = 0)

    xdata = train_X
    yfdata = train_Y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    xp = normalize(xdata[:,0].reshape(-1,1), axis = 0, norm = "max")
    yp = normalize(xdata[:,1].reshape(-1,1), axis = 0, norm = "max")
    #p3 = normalize(xdata[:,2].reshape(-1,1), axis = 0, norm = "max")
    zp = normalize(yfdata, axis = 0, norm = "max")

    xpT = normalize(val_X[:,0].reshape(-1,1), axis = 0, norm = "max")
    ypT = normalize(val_X[:,1].reshape(-1,1), axis = 0, norm = "max")

    xFeed = np.append(xp,yp, axis = 1)

    xFeedT = np.append(xpT,ypT, axis = 1)

    method = MLPRegressor(hidden_layer_sizes = (125,50,25), activation = "tanh", solver = "adam", learning_rate = "adaptive", random_state = 0)
    cloud = prob_cloud.cloud(method, pred_type = "mean")

    xVals = xFeed
    yVals = yfdata
    cloud.fit(xVals,yVals)

    total = cloud.predict(xFeed)
    confidence = cloud.predict_confidence(xFeed, total)
    print("==========================================")
    print(mean_squared_error(total,yfdata))

    totalReal = cloud.predict(xFeedT)
    confidenceReal = cloud.predict_confidence(xFeedT, totalReal)
    print(mean_squared_error(totalReal,val_Y))

    #confs = yfdata - total

    ax.scatter(xp,yp,yfdata.astype(float), label = "True")
    ax.scatter(xp,yp,total.astype(float), label = 0)



    for i in range(3, xdata.shape[1], 2):
        xp = normalize(xdata[:,i-1].reshape(-1,1), axis = 0, norm = "max")
        yp = normalize(xdata[:,i].reshape(-1,1), axis = 0, norm = "max")

        xpT = normalize(val_X[:,i-1].reshape(-1,1), axis = 0, norm = "max")
        ypT = normalize(val_X[:,i].reshape(-1,1), axis = 0, norm = "max")

        xFeed = np.append(xp,yp, axis = 1)
        xFeedT = np.append(xpT,ypT, axis = 1)

        method = MLPRegressor(hidden_layer_sizes = (125,50,25), activation = "tanh", solver = "adam", learning_rate = "adaptive", random_state = 0)
        cloud = prob_cloud.cloud(method, pred_type = "mean")
        cloud.fit(xFeed,yfdata)

        totalTemp = cloud.predict(xFeed)
        confidenceTemp = cloud.predict_confidence(xFeed, totalTemp)

        for j in range(0, total.shape[0]):
            if(confidenceTemp[j] > confidence[j]):
                total[j] = totalTemp[j]
                confidence[j] = confidenceTemp[j]

        print("==========================================")
        print(mean_squared_error(total,yfdata))



        totalTempReal = cloud.predict(xFeedT)
        confidenceTempReal = cloud.predict_confidence(xFeedT, totalTempReal)

        for j in range(0, totalReal.shape[0]):
            if(confidenceTempReal[j] > confidenceReal[j]):
                totalReal[j] = totalTempReal[j]
                confidenceReal[j] = confidenceTempReal[j]

        print(mean_squared_error(totalReal,val_Y))

        #confs = yfdata - total
        #ax.scatter(xp,yp,confs.astype(float), label = i-1)



    xp = normalize(xdata[:,0].reshape(-1,1), axis = 0, norm = "max")
    yp = normalize(xdata[:,1].reshape(-1,1), axis = 0, norm = "max")
    ax.scatter(xp,yp,total.astype(float), label = i-1)





    #ax = fig.add_subplot(111, projection = '3d')


    ax.legend()
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Rates')
    #ax.set_zlabel('Rates')

    plt.show()

if __name__ == "__main__":
  main()
