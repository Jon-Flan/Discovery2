# -*- coding: utf-8 -*-
"""

@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland

"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle



"""
    File 5 
    
    Using the test and train data created from Pre_Processing3.py the data is read in
    and converted to numpy arrays for the machine learning process. Then a mask is applied 
    to each data set incase any nan's have arised from the transformation.
    
    Once that is completed the labels are seperated from the flux reading data and a confirmed planet and
    false reading are extractedf by index for plotting purposes later on.
    
    Beucase there is a 75/25 imbalance in the data weights are calculated for the label sets. Two different 
    versions of each machine learning technique are applied then to see the differences in weighted versus 
    non weighted.
    
    the data is then ready for export into the machine learning algorithms to test with a
    X-Train (Flux Data)
    Y-Train (Labels)
    
    X-Test (Flux Data)
    Y-Test (Labels)
    
"""

# Create Random Seed
np.random.seed(1)


# train and test data and split out into X Train, Y Train, X Test and Y Test
def get_data():
    # Import the data
    df1 = pd.read_csv('D:/College/year_4/semester_1/Software_project/Discovery2/data/Output/train_data.csv')
    df2 = pd.read_csv('D:/College/year_4/semester_1/Software_project/Discovery2/data/Output/test_data.csv')
    df3 = pd.read_csv('D:/College/year_4/semester_1/Software_project/Discovery2/data/Output/val_data.csv')
    
    
    # Create numpy array of each data frame, convert to float
    train_data = np.array(df1, dtype=np.float32)
    test_data = np.array(df2, dtype=np.float32)
    val_data = np.array(df3, dtype=np.float32)
    
    # remove nan data from arrays
    mask_train = np.any(np.isnan(train_data), axis=1)
    mask_test = np.any(np.isnan(test_data), axis=1)
    mask_val = np.any(np.isnan(val_data), axis=1)
    
    train_data = train_data[~mask_train]
    test_data = test_data[~mask_test]
    val_data = val_data[~mask_val]
    
    
    # Separate data 
    ytrain = train_data[:, 0]
    Xtrain = train_data[:, 1:]

    # Seperate labels
    ytest = test_data[:, 0]
    Xtest = test_data[:, 1:]
    
    # Seperate labels
    yval = val_data[:,0]
    Xval = val_data[:,1:]
    
    
    # get index of a exoplanet and non exoplanet by their first label appearance
    planet = (df1.Label.values == 1).argmax()
    non_planet = (df2.Label.values == 0).argmax()
    
    
    # print shape of training X and Y 
    print('Shape of Xtrain:', np.shape(Xtrain), '\nShape of ytrain:', np.shape(ytrain))
    
    return ytrain, Xtrain, ytest, Xtest, yval, Xval, planet, non_planet


# plot lightcurves 
def plot_lightcurve(frame, x, colour, filename, title):

    # plot what the light curve looks like for an exoplanet star
    plt.plot(frame[x], colour)
    plt.title(title)
    plt.xlabel('Time index')
    plt.ylabel('Light intensity')
    
    plt.savefig('../graphs/'+ 
                filename +'.jpg', bbox_inches="tight", dpi=450)
    plt.show()
    plt.clf()
    plt.close()
    
    
# creating weights on labels to help with imbalance
def create_weights_for_labels(Y_train_cnn):
    neg, pos = np.bincount(np.array(Y_train_cnn, dtype=np.int64))
    total = neg + pos
    
    weight_for_0 = (1 / neg)*(total)/2.0
    weight_for_1 = (1 / pos)*(total)/2.0
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    
    return class_weight


def train_data():   
    # Get all data
    data = get_data()
    ytrain = data[0].copy()
    Xtrain = data[1].copy()
    ytest = data[2].copy()
    Xtest = data[3].copy()
    yval = data[4].copy()
    Xval = data[5].copy()
    
    planet = data[6].copy()
    non_planet = data[7].copy()
    
    # Data copies
    Y_train_cnn = ytrain.copy()
    X_train_cnn = Xtrain.copy()
    
    Y_test_cnn = ytest.copy()
    X_test_cnn = Xtest.copy()
    
    Y_val_cnn = yval.copy()
    X_val_cnn = Xval.copy()
    
    # create label weights
    class_weight = create_weights_for_labels(Y_train_cnn)
      
    
    return X_train_cnn, Y_train_cnn, X_test_cnn, Y_test_cnn, X_val_cnn, Y_val_cnn,  planet, non_planet, class_weight



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    