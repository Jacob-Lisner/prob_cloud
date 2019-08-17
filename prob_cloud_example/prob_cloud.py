#Written by Jacob Lisner
#This code is a regressor that attempts to use monte-carlo sampling to directly model
#the probability distribution on a provided data set.



import numpy as np
from sklearn.preprocessing import normalize

class cloud:

    #method: a trainable regressor that will be trained on the point cloud - a MLP with sigmoid activations is recommended
    #for smooth manifolds

    #point_count: the number of points in the point prob_cloud
    #radius: the search radius for data points for the cloud
    #variation: the search radius will vary uniformly between +/- 0.5*variation
    #resolution: when calculating predictions, resolution determines how many points are sampled between 0 and 1 per input
    #pred_type: determines what value from the resolution points is used for the regression: "mode", "median", and "mean"
        #are the options
    #localized: if true, the cloud points will only generate near the fitted points - useful for higher dimensions
    #locaized scale: the edge length of the spawn cube for localized spawning
    #volume_adapt: if true, the denisty value of a given point will consider the specific radius of the point
        #if false, each cloud point only tracks the total number of training points


    def __init__(self, method, point_count = 5000, radius = 0.12, variation = 0.04, resolution = 100, pred_type = "mode", localized = False, localized_scale = 0.2, volume_adapt = False):
        self.method = method
        self.point_count = point_count
        self.radius = radius
        self.variation = variation
        self.resolution = 100


        #Can take forms "best" or "mean", or "median"
        self.pred_type = pred_type
        self.localized = localized
        self.localized_scale = localized_scale
        self.volume_adapt = volume_adapt

    #takes in training data and outputs and oroduces a trained model
    def fit(self, X, y):
        features = X.shape[1]
        samples = X.shape[0]
        pointCount = self.point_count
        radius = self.radius
        variation = self.variation
        method = self.method
        localized = self.localized
        localized_scale = self.localized_scale
        volume_adapt = self.volume_adapt


        xFeed = np.append(X,y.reshape(-1,1), axis = 1)

        #set the scaling factor so that future predictors can be normalized, note that this is currently bugged for multiple calls to fit
        xMax = np.amax(xFeed, axis = 0)
        self.x_max = xMax

        xFeed = normalize(xFeed, axis = 0, norm = "max")


        #generate the point cloud

        points = np.random.rand(pointCount, features+1)
        points = points*(1.0+2.0*radius+2.0*variation) - (radius+variation)
        pointRadii = np.ravel(np.random.rand(pointCount,1)*variation*2.0 + np.ones((pointCount,1))*(radius-variation))

        #calculate the volume to determine local density
        volume = np.ones(pointCount)
        if(volume_adapt):
            volume = np.power(pointRadii/radius,(features+1.0))

        #iterate thru the local points and adjust locations to localize if neccesary
        if(localized):
            for i in range(0, pointCount):
                points[i,:] = (points[i,:]-(0.5*np.ones(features+1)))*localized_scale+xFeed[i%samples,:]


        #deterimine the local densities
        pointVals = np.zeros(pointCount)

        for i in range(0,pointCount):
            total = 0.0
            for j in range(0,samples):
                dist = np.linalg.norm(xFeed[j]-points[i,0:features+1])
                if(dist < pointRadii[i]):
                    total = total + 1.0/volume[i]

            pointVals[i] = total

        self.method.fit(points, pointVals)

    #predicts the y output of the features test values X
    def predict(self, X):

        resolution = self.resolution
        values = X.shape[0]
        features = X.shape[1]

        xScale = self.x_max[0:(self.x_max.shape[0]-1)]
        yScale = self.x_max[self.x_max.shape[0]-1]

        xS = X/xScale

        predictions = np.zeros((values,1))

        for i in range(0, values):
            xFeed = np.zeros((resolution,features+1))
            for j in range(0, resolution):
                xFeed[j,0:features] = xS[i]
                xFeed[j,features] = float(j)/float(resolution)

            yFeed = self.method.predict(xFeed)
            if(self.pred_type == "mean"):
                total = 0.0
                average = 0.0
                for j in range(0, resolution):
                    total = total + yFeed[j]
                    average = average + (float(j)/float(resolution))*yFeed[j]
                predictions[i] = average/total

            elif(self.pred_type == "mode"):
                best = np.argmax(yFeed)
                predictions[i] = float(best)/resolution

            elif(self.pred_type == "median"):
                total = 0.0
                for j in range(0, resolution):
                    total = total + yFeed[j]

                goal = 0
                cumulative = 0.0
                for j in range(0, resolution):
                    cumulative = cumulative + yFeed[j]
                    goal = j
                    if(cumulative > total/2.0):
                        break

                predictions[i] = (float(goal))/resolution



        predictions = predictions*yScale
        return(predictions)

    #takes in a data set X and a suggested output y and
    #outputs the relative confidense (ie. local probability)
    #of y
    def predict_confidence(self, X, y):

        xScale = self.x_max[0:(self.x_max.shape[0]-1)]
        yScale = self.x_max[self.x_max.shape[0]-1]
        resolution = self.resolution
        features = X.shape[1]
        values = X.shape[0]
        xS = X/xScale
        yS = y/yScale

        xFeed = np.append(xS, yS.reshape(-1,1), axis = 1)

        conf = self.method.predict(xFeed).reshape(-1,1)

        xFeed = np.zeros((resolution,features+1))
        for i in range(0, values):
            for j in range(0, resolution):
                xFeed[j,0:features] = xS[i]
                xFeed[j,features] = float(j)/float(resolution)

            yFeed = self.method.predict(xFeed)
            total = 0.0
            for j in range(0, resolution):
                total = total + yFeed[j]

            conf[i] = conf[i]/total


        return(conf)
