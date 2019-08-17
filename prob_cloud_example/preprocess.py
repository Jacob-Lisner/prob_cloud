import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import math
import re
import os

def main():

    #inspired from https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
    directory = "C:/Users/Jacob Lisner/Desktop/cos424_Final/files"

    new = pd.DataFrame()

    map = "2015"

    for filename in os.listdir(directory):

         fileS = str(filename)

         fileS = fileS[0:fileS.find(".")]

         #print(os.path.join(directory, filename))
         data = pd.read_csv(os.path.join(directory, filename),low_memory=False, encoding = " iso-8859-2", delimiter = ",",header = 1, index_col = 1)
         #dataNp = data.to_numpy()
         data.dropna(how = "all", inplace = True)
         data.dropna(how = "all", inplace = True, axis = 1)
         data = data.rename(index = str, columns ={map : fileS})

         if fileS in data.columns:
             cat = data[fileS].copy()
             cat.dropna(how = "all", inplace = True)
             new = pd.concat([new,cat], axis = 1)


    new.to_csv("sets/full_set_" + map + ".csv")
    print("Full:")
    print(new.shape)
    print("First Column Drop:")
    new = new.dropna(thresh = 50, axis = 1)
    print(new.shape)
    print("First Row Drop:")
    new = new.dropna(thresh = 100, axis = 0)
    print(new.shape)
    print("Second Column Drop:")
    new = new.dropna(thresh = new.shape[0]-20, axis = 1)
    print(new.shape)
    #technique from https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-certain-columns-is-nan
    new = new[np.isfinite(new["Suicide rate, female (per 100,000 people)"])]
    #print(new.shape)
    new = new[np.isfinite(new["Suicide rate, male (per 100,000 people)"])]
    #print(new.shape)
    print("Nulls to Impute:")
    print(new.isnull().sum().sum())
    for col in new.columns:
        new[col].fillna(new[col].median(), inplace = True)

    #check from https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
    print(new.isnull().sum().sum())
    new.to_csv("sets/reduced_set_" + map + ".csv")


    female = new["Suicide rate, female (per 100,000 people)"].copy()
    male = new["Suicide rate, male (per 100,000 people)"].copy()

    female.to_csv("sets/female_rate_" + map + ".csv")
    male.to_csv("sets/male_rate_" + map + ".csv")

    new = new.drop(columns = ["Suicide rate, female (per 100,000 people)","Suicide rate, male (per 100,000 people)"])
    print(new.shape)
    new.to_csv("sets/no_suicide_set_" + map + ".csv")
if __name__ == "__main__":
  main()
