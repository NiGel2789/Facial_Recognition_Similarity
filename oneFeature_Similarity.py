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
         opts, args = getopt.getopt(argv,"hf:i:m:k:")
    except getopt.GetoptError:
      print('Usage: oneFeature_Similarity.py -f <folderPath> -i <imagePath> -m <CM/ELBP/HOG> -k <value of k>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('Usage: oneFeature_Similarity.py -f <folderPath> -i <imagePath> -m <CM/ELBP/HOG> -k <value of k>')
      elif opt in ("-f"):
         folderName = arg
      elif opt in ("-i"):
         imageID = arg         
      elif opt in ("-m"):
          model = arg
      else:
          k = arg

    # Decided Model, Moving to getting all images

    path = folderName + '\\'
    files = os.listdir(path)    # list of file names incl. extensions

    # First, find corresponding FD for given image path
    img = cv2.imread(imageID, 0)
    if model == 'CM':
        fd = features.color_moments(img)

    elif model == 'ELBP':
        fd = features.ELBP(img)

    else:
        fd = features.HOG(img)

    # Now, compare with the others, and find the k-Similar Pictures
    kClosest = []
    for file in files:
        if file.endswith(".png"):
            fullPath = path + file
            #outputFile = path + "results_fd.csv" 
            #print(fullPath)
            
            img = cv2.imread(fullPath, 0)

            if model == 'CM':
                cm = features.color_moments(img)
                diff = cm - fd
                diff = np.square(diff)
                diff = np.sum(diff)
                #Check voice memo
            
            elif model == 'ELBP':
                elbp = features.ELBP(img)
                diff = elbp - fd
                diff = np.square(diff)
                diff = np.sum(diff)

            else:
                hog = features.HOG(img)
                diff = hog - fd
                diff = np.square(diff)
                diff = np.sum(diff)

            if diff != 0:
                pair = (file, diff)
                kClosest.append(pair)

    kClosest.sort(key = lambda x: x[1])
    k = int(k)
    for pair in range (0, k):
        imgPathPrint = path + kClosest[pair][0]
        features.process(imgPathPrint, kClosest[pair][0], kClosest[pair][1])
              

if __name__ == "__main__":
   main(sys.argv[1:])
