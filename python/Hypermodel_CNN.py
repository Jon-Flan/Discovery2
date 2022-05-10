# -*- coding: utf-8 -*-
"""

@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland
"""
# Imports
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.utils import shuffle
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt
import matplotlib.pyplot as plt
import Create_training_data as td
from sklearn.metrics import confusion_matrix
from sklearn import metrics as sk_met

# Create Random Seed
np.random.seed(1)


"""
    File 6 
    
    This file is used to tune the hyper parameters of a CNN for accuracy/loss
    
    Using the Keras tuner the model will be built using the basic architecture of
    the paper () using the global view only, with 5 convolutional 
    layers and hyperparams being tested between 16 and 256 filters, stepping up in incriments of 16. 
    
    The convolutional layers will have a kernel size of either 3, 5 or 11, and Relu activation function.
    Max pooling will be used as per the paper with size 5 and strides 2.
    
    4 Dense layers will be added starting with 32 fully connected up to 512 with an an incriment step of 32,
    this will happen on each of the Dense layers being added
    with a drop out of between 0 and 0.5 being tested in incriments of 0.1
    
    final output layer will be a single dense layer of 1 with a sigmoid activation.
    
    For tuning the main evaluation will be done on the validation accuracy.
    
"""

def build_model(hp):
    
    model = keras.Sequential([
            
        #-------------------------------- Adding Convolutional layers --------------------------------#
        
        
        # First Convolutional Layer
        keras.layers.Conv1D(filters = hp.Int('filters Conv_1', min_value = 16, max_value=256, step=16), 
                                  kernel_size = hp.Choice('Kernel Size_1', values=[3,5,11]), 
                                  activation = 'relu',
                                  kernel_regularizer = 'l2', 
                                  padding = 'same',
                                  input_shape = (2000, 1)),
        
        # Pooling layer 1
        keras.layers.MaxPooling1D(pool_size = 5, 
                                  strides = 2),
        
        # Second Convolutional Layer
        keras.layers.Conv1D(filters = hp.Int('filters Conv_2', min_value = 16, max_value=256, step=16), 
                                  kernel_size=hp.Choice('Kernel Size_2', values=[3,5,11]), 
                                  activation='relu'),
        
        # Pooling layer 2
        keras.layers.MaxPooling1D(pool_size = 5, 
                                  strides = 2),
        
        # Third Convolutional Layer
        keras.layers.Conv1D(filters = hp.Int('filters Conv_3', min_value = 16, max_value=256, step=16), 
                                  kernel_size=hp.Choice('Kernel Size_3', values=[3,5,11]), 
                                  activation='relu'),
        # Pooling layer 3
        keras.layers.MaxPooling1D(pool_size = 5, 
                                  strides = 2),
        
        # Fourth Convolutional Layer
        keras.layers.Conv1D(filters = hp.Int('filters Conv_4', min_value = 16, max_value=256, step=16), 
                                  kernel_size=hp.Choice('Kernel Size_4', values=[3,5,11]), 
                                  activation='relu'),
        # Pooling layer 4
        keras.layers.MaxPooling1D(pool_size = 5, 
                                  strides = 2),
        
        # Fifth Convolutional Layer
        keras.layers.Conv1D(filters = hp.Int('filters Conv_5', min_value = 16, max_value=256, step=16), 
                                  kernel_size=hp.Choice('Kernel Size_5', values=[3,5,11]), 
                                  activation='relu'),
        # Pooling layer 5
        keras.layers.MaxPooling1D(pool_size = 5, 
                                  strides = 2),
        
        # Add a flattening layer
        keras.layers.Flatten(),
        
        #---------------------------- Adding Fully connected layers ------------------------------#
        
        # First Dense layer
        keras.layers.Dense(units=hp.Int('Hidden Size_1', min_value=32, max_value=512, step=32), 
                           activation='relu'),
        
        # Drop out for first fully connected layer
        #keras.layers.Dropout(rate=hp.Float('Dropout_1', min_value=0, max_value=0.5, step=0.1 )),
        
        
        # Second Dense layer
        keras.layers.Dense(units=hp.Int('Hidden Size_2', min_value=32, max_value=512, step=32), 
                           activation='relu'),
        
        # Drop out for second fully connected layer
        #keras.layers.Dropout(rate=hp.Float('Dropout_2', min_value=0, max_value=0.5, step=0.1 )),
        
        
        # Third Dense layer
        keras.layers.Dense(units=hp.Int('Hidden Size_3', min_value=32, max_value=512, step=32), 
                           activation='relu'),
        
        # Drop out for third fully connected layer
        #keras.layers.Dropout(rate=hp.Float('Dropout_3', min_value=0, max_value=0.5, step=0.1 )),
        
        
        # Fourth Dense layer
        keras.layers.Dense(units=hp.Int('Hidden Size_4', min_value=32, max_value=512, step=32), 
                           activation='relu'),
        
        # Drop out for fourth fully connected layer
        #keras.layers.Dropout(rate=hp.Float('Dropout_4', min_value=0, max_value=0.5, step=0.1 )),
        
        
        #--------------------------------- Adding Output layer -----------------------------------#
        
        keras.layers.Dense(1, activation="sigmoid")
        
        ])
    
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.BinaryCrossentropy(
                                                        from_logits=False,
                                                        label_smoothing=0.0,
                                                        axis=-1,
                                                        reduction="auto",
                                                        name="binary_crossentropy",),
                metrics=['accuracy'])
    
    return model


# run and tune the model using hyperband algorithm
def tune_hyper_params(X_train_cnn, Y_train_cnn, x_val, y_val):
    
    # parameters for tunning
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50, #  as used in the paper
        factor = 3,
        directory = 'Tuned_CNN',
        project_name = 'Discovery2',
        hyperband_iterations=2)
    
    # run the model and search for best hyperparameters according to accuracy
    tuner.search(X_train_cnn,
                 Y_train_cnn,
                 epochs=50,# as used in the paper
                 batch_size = 64,
                 validation_data=(x_val, y_val))
    
    # return the best model and best parameters
    return tuner


def run_tuner():
    training_data = td.train_data()

    X_train_cnn = training_data[0]
    Y_train_cnn = training_data[1]
    
    X_val_cnn = training_data[4]
    Y_val_cnn = training_data[5]
    
    # apply gaussian filter
    X_train_cnn = ndimage.filters.gaussian_filter(X_train_cnn, sigma=50)
    X_val_cnn = ndimage.filters.gaussian_filter(X_val_cnn, sigma=50)
    
    # apply min max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_cnn = scaler.fit_transform(X_train_cnn)
    X_val_cnn = scaler.fit_transform(X_val_cnn)

    # shuffle dataset
    X_train_cnn, Y_train_cnn = shuffle(X_train_cnn, Y_train_cnn)
    X_val_cnn, Y_val_cnn = shuffle(X_val_cnn, Y_val_cnn)
    
    # Reshape and shuffle the data to be used in building the CNN
    X_train_cnn = np.expand_dims(X_train_cnn, axis=2)
    X_val_cnn = np.expand_dims(X_val_cnn, axis=2)
    

    # run the model tuner to fine tune hyper parameters                           
    tuner = tune_hyper_params(X_train_cnn, Y_train_cnn, X_val_cnn, Y_val_cnn)
    
    
    return tuner


# run the model tuner to get best hyperparams
def create_model():
    # run the tuner
    tuner = run_tuner()
    
    # get the best hyper parameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    
    # build the model from the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    
    return model, best_hps, tuner


# testing the model again to get the best epoch
def test_hypermodel(model):
    
    # call the test and validation data
    training_data = td.train_data()

    X_train_cnn = training_data[0]
    Y_train_cnn = training_data[1]
    
    X_test_cnn = training_data[2]
    Y_test_cnn = training_data[3]
    
    X_val_cnn = training_data[4]
    Y_val_cnn = training_data[5]
    
    # apply gaussian filter
    X_train_cnn = ndimage.filters.gaussian_filter(X_train_cnn, sigma=50)
    X_test_cnn = ndimage.filters.gaussian_filter(X_test_cnn, sigma=50)
    X_val_cnn = ndimage.filters.gaussian_filter(X_val_cnn, sigma=50)
    
    # apply min max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_cnn = scaler.fit_transform(X_train_cnn)
    X_test_cnn = scaler.fit_transform(X_test_cnn)
    X_val_cnn = scaler.fit_transform(X_val_cnn)
    
    # shuffle the data set
    X_train_cnn, Y_train_cnn = shuffle(X_train_cnn, Y_train_cnn)
    X_test_cnn, Y_test_cnn = shuffle(X_test_cnn, Y_test_cnn)
    X_val_cnn, Y_val_cnn = shuffle(X_val_cnn, Y_val_cnn)
    
    
    # Reshape  to be used in testing the CNN
    X_train_cnn = np.expand_dims(X_train_cnn, axis=2)
    X_test_cnn = np.expand_dims(X_test_cnn, axis=2)
    X_val_cnn = np.expand_dims(X_val_cnn, axis=2)
    
    
    # set the stop early callback
    stop_early = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    
    # test the model with the training data
    history = model.fit(X_train_cnn, 
                        Y_train_cnn, 
                        epochs = 50, 
                        batch_size=64,
                        validation_data=(X_val_cnn, Y_val_cnn),
                        callbacks=[stop_early])    
    
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    
    
    return best_epoch


def create_hypermodel(model, best_epoch, best_hps):
    
    # get the testing data
    training_data = td.train_data()
    
    X_train_cnn = training_data[0]
    Y_train_cnn = training_data[1]
    
    X_test_cnn = training_data[2]
    Y_test_cnn = training_data[3]
    
    X_val_cnn = training_data[4]
    Y_val_cnn = training_data[5]
    
    # apply gaussian filter
    X_train_cnn = ndimage.filters.gaussian_filter(X_train_cnn, sigma=50)
    X_test_cnn = ndimage.filters.gaussian_filter(X_test_cnn, sigma=50)
    X_val_cnn = ndimage.filters.gaussian_filter(X_val_cnn, sigma=50)
    
    # apply min max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_cnn = scaler.fit_transform(X_train_cnn)
    X_test_cnn = scaler.fit_transform(X_test_cnn)
    X_val_cnn = scaler.fit_transform(X_val_cnn)
    
    # shuffle the data set
    X_train_cnn, Y_train_cnn = shuffle(X_train_cnn, Y_train_cnn)
    X_test_cnn, Y_test_cnn = shuffle(X_test_cnn, Y_test_cnn)
    X_val_cnn, Y_val_cnn = shuffle(X_val_cnn, Y_val_cnn)
    
    
    # Reshape  to be used in testing the CNN
    X_train_cnn = np.expand_dims(X_train_cnn, axis=2)
    X_test_cnn = np.expand_dims(X_test_cnn, axis=2)
    X_val_cnn = np.expand_dims(X_val_cnn, axis=2)
    
    # create hyper model and evalute performance
    new_model = model.hypermodel.build(best_hps)
    new_model.summary()
    new_model_history = new_model.fit(X_train_cnn, 
                       Y_train_cnn, 
                       epochs = best_epoch, 
                       batch_size=64,
                       validation_data=(X_val_cnn, Y_val_cnn))
    
    # evaluate the results
    eval_result = new_model.evaluate(X_test_cnn, Y_test_cnn)
    print("[test loss, test accuracy]:", eval_result)
    
    
    return new_model


# main function to return tuned hypermodel cnn
def main():
    # create the model and return values
    model_output = create_model()
    model = model_output[0]
    best_hps = model_output[1]
    tuner = model_output[2]
    
    # test the hyper model and get best epochs
    best_epoch = test_hypermodel(model)
    
    cnn_hypermodel = create_hypermodel(tuner, best_epoch, best_hps)
    

    return cnn_hypermodel
    

# calling main function.
if __name__ == "__main__":
    
    cnn_hypermodel = main()
    
    # Save Created model.
    #cnn_hypermodel.save("./Models/CNN_Hypermodel.h5")

    