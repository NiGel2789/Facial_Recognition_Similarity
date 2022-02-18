# Facial Recognition and Similarity System
A simple facial recognition system that given a feature model (color moments, extended linear binary patterns, histogram of gradients) and an image from the dataset, gives the top-k most similar faces and their similarity scores. There is also an option to base facial similarity on all three models combined.

Technologies Used: Python, skimage, sklearn, pandas, matplotlib, NumPy
Dataset: Olivetti Faces Dataset

Files:
oneFeature_Similarity - Calculating similarity based on one of the features (CM, ELBP, HOG)
allFeatures_Similarity - Calculating similarity based on all features
features.py - Methods for calculating the features

Date completed: 9/15/2021
