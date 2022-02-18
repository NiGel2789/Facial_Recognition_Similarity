import cv2
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

def color_moments(imageID):
    window_shape = (8, 8)
    row = 0
    col = 0
    fd = np.empty((0, 3), float)

    # Splits the picture into 8x8 windows for processing
    split = view_as_windows(imageID, window_shape)

    for row in range (0, 64, 8):
        for col in range (0, 64, 8):
            sum = np.sum(split[row,col])
            
            # Mean
            mean = sum/64

            # Standard Deviation
            sd = np.sqrt(np.sum(np.square(np.subtract(split[row, col], mean)))/64)

            # Skew
            skew = np.cbrt(np.sum(np.power(np.subtract(split[row, col], mean), 3))/64)

            descriptor = np.array([[mean, sd, skew]])
            fd = np.concatenate((fd, descriptor), axis=0)

    return fd

def ELBP(imageID):
    radius = 3
    n_points = 8*radius

    # Extended Local Binary Pattern - Grayscale Rotational Invariance
    elbp = local_binary_pattern(imageID, n_points, radius, method='ror')

    return elbp

# Histogram of Gradients
def HOG(imageID):
    feature_descriptors, hog_image = hog(imageID, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys', feature_vector=True)

    return feature_descriptors

def process(filename, name, value) -> None:
    image = mpimg.imread(filename, 0)
    plt.gray()
    plt.imshow(image)
    print(value)
    plt.title(value,fontsize=16)
    plt.suptitle(name,fontsize=24, y=1)
    plt.show()
