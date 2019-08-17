#Written by Jacob Lisner


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
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
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


    #read in the data
    map = "2015"

    clusterData = pd.read_csv("sets/reduced_set_" + map + ".csv")
    xData = pd.read_csv("sets/no_suicide_set_" + map + ".csv")
    yFData = pd.read_csv("sets/female_rate_" + map + ".csv", header = None)
    #yMData = pd.read_csv("sets/male_rate_" + map + ".csv", header = None)


    xlabels = xData.values[:,0]
    yflabels = yFData.values[:,0]
    #ymlabels = yMData.values[:,0]
    flabels = xData.columns.values[1:]

    xdata = xData.values[:,1:]
    yfdata = yFData.values[:,1:]
    #ymdata = yMData.values[:,1:]

    features = xdata.shape[1]
    samples = xdata.shape[0]



    #find and print the two best national metrics
    scoreBest1 = 0
    indexBest1 = 0
    scoreBest2 = 0
    indexBest2 = 0

    for i in range(0,features):
        method = LinearRegression()
        method.fit(xdata[:,i].reshape(-1,1),yfdata)

        score = method.score(xdata[:,i].reshape(-1,1),yfdata)
        if(score > scoreBest2):
            if(score > scoreBest1):
                scoreBest2 = scoreBest1
                indexBest2 = indexBest1
                scoreBest1 = score
                indexBest1 = i
            else:
                scoreBest2 = score
                indexBest2 = i

    print("======================")
    print("BEST MARGINAL FEATURES (By R Value):")
    print(flabels[indexBest1])

    print(flabels[indexBest2])


    #x = feature 1, y = feature 2, z = suicide rate


    xp = normalize(xdata[:,indexBest1].reshape(-1,1), axis = 0, norm = "max")
    yp = normalize(xdata[:,indexBest2].reshape(-1,1), axis = 0, norm = "max")
    zp = normalize(yfdata, axis = 0, norm = "max")

    p0 = normalize(xdata[:,0].reshape(-1,1), axis = 0, norm = "max")


    #xFeed is the actual features in the data set (with yFeed being the predicted values)
    xFeed = np.append(xp,yp, axis = 1)
    xFeed = np.append(xFeed,zp, axis = 1)

    #For demonstration, produce a sample point cloud and color code the densities of the points

    pointCount = 5000
    radius = 0.12
    variation = 0.06
    points = np.random.rand(pointCount,3)
    points = points*(1.0+2.0*radius+2.0*variation) - (radius+variation)
    pointRadii = np.ravel(np.random.rand(pointCount,1)*variation*2.0 + np.ones((pointCount,1))*(radius-variation))
    pointVals = np.zeros(pointCount)

    for i in range(0,pointCount):
        total = 0.0
        for j in range(0,samples):
            dist = np.linalg.norm(xFeed[j]-points[i,0:3])
            if(dist < pointRadii[i]):
                total = total + 1

        pointVals[i] = total


    #For demonstration, train an MLP on the point cloud

    mlpPrime = MLPRegressor(hidden_layer_sizes = (200,75,50), activation = "tanh", solver = "adam", learning_rate = "adaptive", random_state = 0)
    mlpPrime.fit(points,pointVals)
    #print(mlpPrime.score(points,pointVals))


    #plot the probability cloud with the calculated densities
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xp,yp,zp)
    ax.scatter(points[:,0],points[:,1],points[:,2], cmap = "hot", c = np.minimum(250.0*np.ones(pointCount),0.03*pointVals/(pointRadii*pointRadii*pointRadii)))
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Suicide Rate')
    ax.set_title('Point Cloud (Lighter = More Neighbors, Blue = Original)')


    resolution = 10
    pointCheck = np.zeros(((resolution+1)*(resolution+1)*(resolution+1),3))
    for i in range(0,resolution+1):
        for j in range(0, resolution+1):
            for k in range(0, resolution+1):
                pointCheck[i+(resolution+1)*j+(resolution+1)*(resolution+1)*k,0] = float(i)/float(resolution)
                pointCheck[i+(resolution+1)*j+(resolution+1)*(resolution+1)*k,1] = float(j)/float(resolution)
                pointCheck[i+(resolution+1)*j+(resolution+1)*(resolution+1)*k,2] = float(k)/float(resolution)

    pointCheckVals = mlpPrime.predict(pointCheck)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xp,yp,zp)
    ax.scatter(pointCheck[:,0],pointCheck[:,1],pointCheck[:,2], cmap = "hot", c = np.minimum(250.0*np.ones((resolution+1)*(resolution+1)*(resolution+1)),0.07*pointCheckVals/(radius*radius*radius)))
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Suicide Rate')
    ax.set_title('Probability Distribution (Lighter = Higher Prob, Blue = Original)')


    print("Testing")
    method = MLPRegressor(hidden_layer_sizes = (200,70,50), activation = "tanh", solver = "adam", learning_rate = "adaptive", random_state = 0)
    cloud = prob_cloud.cloud(method, point_count = 5000, radius = 0.06, variation = 0.02)



    xVals = np.append(xdata[:,indexBest1].reshape(-1,1),xdata[:,indexBest2].reshape(-1,1), axis = 1)
    yVals = yfdata
    cloud.fit(xVals,yVals)
    newYs = cloud.predict(xVals)
    cloud.pred_type = "mean"
    newYs2 = cloud.predict(xVals)
    cloud.pred_type = "median"
    newYs3 = cloud.predict(xVals)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xp,yp,yfdata.astype(float), label = "Orignial")
    ax.scatter(xp,yp,newYs, label = "Mode")
    ax.scatter(xp,yp,newYs2, label = "Mean")
    ax.scatter(xp,yp,newYs3, label = "Median")
    ax.legend()
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Suicide Rate')
    ax.set_title('Regression Values')

    print("Error Values using the demonstration cloud")
    print("Mode Error (MSE):")
    print(mean_squared_error(newYs,yfdata.astype(float)))
    print("Mean Error (MSE):")
    print(mean_squared_error(newYs2,yfdata.astype(float)))
    print("Median Error (MSE):")
    print(mean_squared_error(newYs3,yfdata.astype(float)))

    #xVals = xdata
    #yVals = yfdata
    method = MLPRegressor(hidden_layer_sizes = (200,70,50), activation = "tanh", solver = "adam", learning_rate = "adaptive", random_state = 0)
    cloud = prob_cloud.cloud(method, point_count = 5000, radius = 0.06, variation = 0.02)
    cloud.fit(xVals,yVals)
    newYs = cloud.predict(xVals)

    confs = cloud.predict_confidence(xVals,yVals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(yVals,confs, p0)
    ax.set_xlabel('Suicide Rate')
    ax.set_ylabel('Predicted Rate')
    ax.set_ylabel('New Feature')
    ax.set_title("Sample of next iteration")

    #plt.show()

    cloud.pred_type = "mean"
    newYs2 = cloud.predict(xVals)
    cloud.pred_type = "median"
    newYs3 = cloud.predict(xVals)

    print("Errors using prob_cloud")
    print("Mode Error (MSE):")
    print(mean_squared_error(newYs,yfdata.astype(float)))
    print("Mean Error (MSE):")
    print(mean_squared_error(newYs2,yfdata.astype(float)))
    print("Median Error (MSE):")
    print(mean_squared_error(newYs3,yfdata.astype(float)))

    print("======================")
    #print("Hold Out Ratings:")


    #inspired from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html


    #The following is a 5 fold validation of the the density cloud predictor, with
    #the first entry being the "mode", the second "mean", and the third "median"
    #UNCOMMENT BELLOW FOR VALIDATION:
    #========================================================================================

    #folds = 5
    #kf = KFold(n_splits=folds, shuffle = True, random_state = 0)
    #kf.get_n_splits(xVals)

    #errors = np.zeros((folds,3))
    #errorI = 0
    #for train_index, test_index in kf.split(xVals):

    #   X_train, X_test = xVals[train_index], xVals[test_index]
    #   y_train, y_test = yVals[train_index], yVals[test_index]

    #   method = MLPRegressor(hidden_layer_sizes = (200,70,50), activation = "tanh", solver = "adam", learning_rate = "adaptive", random_state = 0)
    #   cloud = prob_cloud.cloud(method, point_count = 50000, radius = 0.03, variation = 0.01)
    #   cloud.fit(X_train,y_train)
    #   newYs = cloud.predict(X_test)
    #   cloud.pred_type = "mean"
    #   newYs2 = cloud.predict(X_test)
    #   cloud.pred_type = "median"
    #   newYs3 = cloud.predict(X_test)
    #   errors[errorI, 0] = mean_squared_error(newYs,y_test.astype(float))
    #   errors[errorI, 1] = mean_squared_error(newYs2,y_test.astype(float))
    #   errors[errorI, 2] = mean_squared_error(newYs3,y_test.astype(float))
    #   print(errorI)
    #   errorI = errorI+1

    #print(errors)
    #print(np.average(errors, axis = 0))

    #========================================================================================


    plt.show()

if __name__ == "__main__":
  main()
