# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:17:39 2019

@author: Colin Cumming
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob 
import cv2
import os 
from keras.models import Sequential
from keras.layers.core import Dense

def load_house_attributes(path):
    # define column names to ensure proper titles on document
    columns = ["price", "neighbourhood", "POI"]
    df = pd.read_csv(path, sep =",", header=0, names=columns)
    return df

def process_house_attributes(df, train, test):
    # onehot encode categorical data
    oneHot = LabelBinarizer().fit(df["neighbourhood"])
    trainCategory = oneHot.transform(train["neighbourhood"])
    testCategory = oneHot.transform(test["neighbourhood"])
    
    # scale POI to 0 to 1, Nic might have already done this
    cs = MinMaxScaler()
    trainCont = cs.fit_transform(train["POI"])
    testCont = cs.fit_transform(train["POI"])
    
    # concatenate new columns together
    trainX = np.hstack([trainCategory, trainCont])
    testX= np.hstack([testCategory, testCont])
    
    return(trainX, testX)

def load_house_images(df, inputPath):
	images = []

	for i in df.index.values: 
		basePath = os.path.sep.join([inputPath, "{}_*".format(i+1)])
		print(basePath)
		housePaths = sorted(list(glob.glob(basePath)))

		inputImages = []
		outputImage = np.zeros((64,32,3) , dtype ="uint8")
		for housePath in housePaths:
			image = cv2.imread(housePath)
			image = cv2.resize(image,(32,32))
			inputImages.append(image)
		print(len(inputImages))
		outputImage[0:32, 0:32] = inputImages[0]
		outputImage[32:64, 0:32] = inputImages[1]

		images.append(outputImage)
        
    return np.array(images)