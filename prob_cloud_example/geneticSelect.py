import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor


#uses a genetic algorithm to select optimal features to predict with using a RandomForestRegressor. Returns
#an array with the indices of the optimal features
#X - the input data
#y - the output vector
#n_features: the number of features to return
#pop_size: the number of features sets to test
#n_gen: the number of generations for the genetic algorithm
#cuttoff: the proportion of total features that will recombine
#mut: the proportion of the total features that will have mutations, must be less than 1-cuttoff
#safe: the proportion of total features that will be perfectly preserved: must be greater than cuttoff
def geneticSelect(X, y, n_features, pop_size = 100, n_gen = 40, cuttoff = 0.5, mut = 0.1, safe = 0.05):

    individuals = np.random.randint(X.shape[1], size = (pop_size,n_features))
    score = np.zeros((pop_size,1))

    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = 0.2, random_state = 0)

    #int version of cuttoff
    cI = int(pop_size*cuttoff)
    if(cI%2 == 1):
        cI = cI + 1

    #int version of mut
    mI = int(pop_size*mut)
    #int version of safe
    sI = int(pop_size*safe)


    if(pop_size-mI < sI):
        return("Error: Not Enough Mutation Points")


    best = 0

    #iterate thru the feature combinations and calculate the fitness
    for k in range(0, n_gen):
        for i in range(0, pop_size):
            features = individuals[i]
            xInd = np.zeros((train_X.shape[0],n_features))
            testInd = np.zeros((test_X.shape[0],n_features))
            for j in range(0, n_features):
                xInd[:,j] = train_X[:,features[j]]
                testInd[:,j] = test_X[:,features[j]]

            #calculate fitness
            regL = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=3).fit(xInd,np.ravel(train_Y))
            score[i,:] = regL.score(testInd, test_Y)
            if(score[i,:] > best):
                best = score[i,:]

        set = np.concatenate((individuals, score), axis = 1)

        #inspired by https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
        #sorts by fitness
        set = np.flip(set[set[:,n_features].argsort()], axis = 0)

        #if converged, return
        if(k == n_gen-1):
            return set[0, 0:n_features].astype(int), set[0,n_features]

        set = set[:,0:n_features]
        set = set[0:cI, :]

        #calculate the new population
        setCross = np.zeros((pop_size-cI,n_features))
        for i in range(0, setCross.shape[0]):
            parents = np.random.randint(cI, size = 2)
            which = np.random.randint(2, size = n_features)

            for j in range(0, setCross.shape[1]):
                setCross[i,j] = set[parents[which[j]],j]

        individuals = np.concatenate((set,setCross), axis = 0).astype(int)

        mutInd = np.random.randint(sI, pop_size, size = mI)
        for i in range(0, mI):
            j = random.randint(0,n_features-1)
            jVal = random.randint(0,X.shape[1]-1)
            individuals[mutInd[i],j] = jVal


        #print(best)
        #print(individuals)
