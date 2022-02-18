import features

import sys, getopt
import cv2
import os
import csv 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from skimage.util.shape import view_as_windows
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Getting Olivetti Set
data = fetch_olivetti_faces()

# Storing data as ndarray
images = data.images
target = data.target

def main(argv):
    inputfile = ''
    try:
         opts, args = getopt.getopt(argv,"hf:i:k:")
    except getopt.GetoptError:
      print('Usage: allFeatures_Similarity.py -f <folderPath> -i <imagePath> -k <value of k>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('Usage: allFeatures_Similarity.py -f <folderPath> -i <imagePath> -k <value of k>')
      elif opt in ("-f"):
         folderName = arg
      elif opt in ("-i"):
         imageID = arg         
      else:
          k = arg

    # Moving to getting all images

    path = folderName + '\\'
    files = os.listdir(path)    # list of file names incl. extensions

    # First, find corresponding FD for given image path
    img = cv2.imread(imageID, 0)

    # CM8*8
    cm = features.color_moments(img)

    # ELBP
    elbp = features.ELBP(img)

    # HOG
    hog = features.HOG(img)

    # Now, compare with the others, and find the k-Similar Pictures
    kClosest = []
    cmRank = []
    elbpRank = []
    hogRank = []
    count = 0
    for file in files:
        if file.endswith(".png"):
            count = count + 1
            fullPath = path + file
            #outputFile = path + "results_fd.csv" 
            #print(fullPath)
            
            img = cv2.imread(fullPath, 0)

            #CM8*8
            cm2 = features.color_moments(img)
            diffCM = cm2 - cm
            diffCM = np.square(diffCM)
            diffCM = np.sum(diffCM)
            pairCM = (file, diffCM)
            cmRank.append(pairCM)

            #ELBP
            elbp2 = features.ELBP(img)
            diffELBP = elbp2 - elbp
            diffELBP = np.square(diffELBP)
            diffELBP = np.sum(diffELBP)
            pairELBP = (file, diffELBP)
            elbpRank.append(pairELBP)

            # HOG
            hog2 = features.HOG(img)
            diffHOG = hog2 - hog
            diffHOG = np.square(diffHOG)
            diffHOG = np.sum(diffHOG)
            pairHOG = (file, diffHOG)
            hogRank.append(pairHOG)

    # All populated, find max for each model
    maxCM = max(cmRank,key=lambda item:item[1])[1]
    maxELBP = max(elbpRank,key=lambda item:item[1])[1]
    maxHOG = max(hogRank,key=lambda item:item[1])[1]

    # Normalize each ranking by diving w maximum
    for each in range(0, count):
        normCM = cmRank[each][1]/maxCM
        normELBP = elbpRank[each][1]/maxELBP
        normHOG = hogRank[each][1]/maxHOG

        diff = normCM + normELBP + normHOG

        if diff != 0:
            pair = (cmRank[each][0], diff)
            kClosest.append(pair)

    # Sort by value    
    kClosest.sort(key = lambda x: x[1])
    k = int(k)

    for pair in range (0, k):
        imgPathPrint = path + kClosest[pair][0]
        features.process(imgPathPrint, kClosest[pair][0], kClosest[pair][1])
              

if __name__ == "__main__":
   main(sys.argv[1:])
