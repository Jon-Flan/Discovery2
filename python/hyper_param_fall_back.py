# -*- coding: utf-8 -*-
"""

@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland
"""
# Imports
import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle
import keras_tuner as kt

import Create_training_data as td

# Create Random Seed
np.random.seed(1)


"""
    File 6 
    
    This file is used to tune the hyper parameters of a CNN for accuracy/loss
    
    Using the Keras tuner the model will be built with 3 convolutional 
    layers and between 16 and 256 filters, stepping up in incriments of 16. 
    
    The convolutional layers will have a kernel size of either 3, 5 or 11, and Relu activation function.
    Max and average pooling will be checked for each layer.
    
    3 Dense layers will be added starting with 32 fully connected up to 128 with an an incriment step of 10,
    this will happen on each of the Dense layers being added
    with a drop out of between 0 and 0.5 being tested in incriments of 0.1
    
    final output layer will be a dense layer of 2 with a sigmoid activation 
    
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
        keras.layers.MaxPooling1D(pool_size = hp.Choice('Pool Size 1', values=[2,4]), 
                                  strides = hp.Choice('Stride Size 1', values=[2,4])),
        
        # Second Convolutional Layer
        keras.layers.Conv1D(filters = hp.Int('filters Conv_2', min_value = 16, max_value=256, step=16), 
                                  kernel_size=hp.Choice('Kernel Size_2', values=[3,5,11]), 
                                  activation='relu'),
        
        # Pooling layer 2
        keras.layers.MaxPooling1D(pool_size = hp.Choice('Pool Size 2', values=[2,4]), 
                                  strides = hp.Choice('Stride Size 2', values=[2,4])),
        
        # Third Convolutional Layer
        keras.layers.Conv1D(filters = hp.Int('filters Conv_3', min_value = 16, max_value=256, step=16), 
                                  kernel_size=hp.Choice('Kernel Size_3', values=[3,5,11]), 
                                  activation='relu'),
        # Pooling layer 3
        keras.layers.MaxPooling1D(pool_size = hp.Choice('Pool Size 3', values=[2,4]), 
                                  strides = hp.Choice('Stride Size 3', values=[2,4])),
        
        # Add a flattening layer
        keras.layers.Flatten(),
        
        #---------------------------- Adding Fully connected layers ------------------------------#
        
        # First Dense layer
        keras.layers.Dense(units=hp.Int('Hidden Size_1', min_value=32, max_value=128, step=32), 
                           activation='relu'),
        
        # Drop out for first fully connected layer
        keras.layers.Dropout(rate=hp.Float('Dropout_1', min_value=0, max_value=0.5, step=0.1 )),
        
        
        # Second Dense layer
        keras.layers.Dense(units=hp.Int('Hidden Size_2', min_value=32, max_value=128, step=32), 
                           activation='relu'),
        
        # Drop out for second fully connected layer
        keras.layers.Dropout(rate=hp.Float('Dropout_2', min_value=0, max_value=0.5, step=0.1 )),
        
        
        # Third Dense layer
        keras.layers.Dense(units=hp.Int('Hidden Size_3', min_value=32, max_value=128, step=32), 
                           activation='relu'),
        
        # Drop out for third fully connected layer
        keras.layers.Dropout(rate=hp.Float('Dropout_3', min_value=0, max_value=0.5, step=0.1 )),
        
        
        
        #--------------------------------- Adding Output layer -----------------------------------#
        
        keras.layers.Dense(2, activation="sigmoid")
        
        ])
    
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    return model

    
    
    

              
# =============================================================================
# # Parameteres for building the CNN model
# def build_model(hp):
#   # input shape 
#   inputs = keras.Input(shape=(2000, 1))
#   x = inputs
#   
#   # range of convolutional layers
#   for i in range(hp.Int('conv_blocks', 1, 4, default=1)):
#     filters = hp.Int('filters_' + str(i), 16, 256, step=16)
#     kernel = hp.Int('kernel_' +str(i), 3, 11, step=2)
#     for _ in range(2):
#       x = keras.layers.Convolution1D(
#         filters, kernel_size=kernel, padding='same', activation="relu")(x)
#     if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
#       x = keras.layers.MaxPool1D(pool_size = 2, strides = 2)(x)
#     else:
#       x = keras.layers.AvgPool1D(pool_size = 2, strides = 2)(x)
#       
#   # adding 3 fully connected layers and dropout
#   x = keras.layers.GlobalAvgPool1D()(x)
#   x = keras.layers.Dense(
#       hp.Int('hidden_size_0', 30, 100, step=10, default=50),
#       activation='relu')(x)
#   x = keras.layers.Dropout(
#       hp.Float('dropout_0', 0, 0.5, step=0.1, default=0.5))(x)
#   x = keras.layers.Dense(
#       hp.Int('hidden_size_1', 30, 100, step=10, default=50),
#       activation='relu')(x)
#   x = keras.layers.Dropout(
#       hp.Float('dropout_1', 0, 0.5, step=0.1, default=0.5))(x)
#   x = keras.layers.Dense(
#       hp.Int('hidden_size_2', 30, 100, step=10, default=50),
#       activation='relu')(x)
#   x = keras.layers.Dropout(
#       hp.Float('dropout_2', 0, 0.5, step=0.1, default=0.5))(x)
#   
#   # output layer of two nodes with sigmoid activation
#   outputs = keras.layers.Dense(2, activation='sigmoid')(x)
#   
#   # compile the model, testing the learning rate and measuring for accuracy
#   model = keras.Model(inputs, outputs)
#   model.compile(
#     optimizer=keras.optimizers.Adam(
#       hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
#     loss='sparse_categorical_crossentropy', 
#     metrics=['accuracy'])
#   
#   return model
# =============================================================================


# run and tune the model using hyperband algorithm
def tune_hyper_params(X_train_cnn, Y_train_cnn, x_val, y_val):
    
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    
    # parameters for tunning
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=30,
        factor = 3,
        directory = 'Params_CNN',
        project_name = 'Discovery2',
        hyperband_iterations=2)
    
    # run the model and search for best hyperparameters according to accuracy
    tuner.search(X_train_cnn,
                 Y_train_cnn,
                 epochs=30,
                 batch_size = 64,
                 validation_data=(x_val, y_val),
                 callbacks=[stop_early])
    
    # return the best model and best parameters
    return tuner.get_best_models(1)[0], tuner.get_best_hyperparameters(1)[0]


def main():
    training_data = td.train_data()

    X_train_cnn = training_data[0]
    Y_train_cnn = training_data[1]
    
    X_val_cnn = training_data[4]
    Y_val_cnn = training_data[5]

    
    # Reshape and shuffle the data to be used in building the CNN
    X_train_cnn = np.expand_dims(X_train_cnn, axis=2)
    X_train_cnn, Y_train_cnn = shuffle(X_train_cnn, Y_train_cnn)
    
    X_val_cnn = np.expand_dims(X_val_cnn, axis=2)
    X_val_cnn, Y_val_cnn = shuffle(X_val_cnn, Y_val_cnn)
    
    # run the model tuner to fine tune hyper parameters                           
    model_info = tune_hyper_params(X_train_cnn, Y_train_cnn, X_val_cnn, Y_val_cnn)
    
    # model information
    best_model = model_info[0]
    best_hyperparameters = model_info[1]
    
    return best_model, best_hyperparameters


# calling main function.
if __name__ == "__main__":
    model_info = main()
    
    best_model = model_info[0]
    best_hyperparameters = model_info[1]
    
    print(best_model.summary())
    
    
    
    
    