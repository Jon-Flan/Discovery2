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
from tensorflow import keras
from sklearn import metrics as sk_met
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from scipy.fftpack import fft
from sklearn.preprocessing import normalize
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from keras import callbacks

import Create_training_data as td

"""
    File 7 
    
    This file is used to ......
    
"""

# Create Random Seed
np.random.seed(1)







# Create a CNN
def create_model(X_samp, activation='relu', learn_rate=0.001):
    # set model to sequential 
    model = keras.Sequential()
    
    # First Convolutional Layer
    model.add(keras.layers.Conv1D(filters=16, input_shape=(X_samp.shape[1], 1), 
                                  kernel_size=(5), 
                                  activation=activation,
                                  kernel_regularizer='l2', 
                                  padding='same', name = 'Conv1D-1'))
    model.add(keras.layers.MaxPooling1D(pool_size = 5, strides = 2, padding='valid', name = 'Maxpool-1'))
 
    # Second Convolutional Layer
    model.add(keras.layers.Conv1D(filters=32, 
                                  kernel_size=(5), 
                                  activation=activation, 
                                  padding='same', name = 'Conv1D-2'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size = 5, strides = 2, padding='valid', name = 'Maxpool-2'))
    
    # Third Convolutional Layer
    model.add(keras.layers.Conv1D(filters=64, 
                                  kernel_size=(5), 
                                  activation=activation, 
                                  padding='same', name = 'Conv1D-3'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size = 5, strides = 2, padding='valid', name = 'Maxpool-3'))
    
    # Fourth Convolutional Layer
    model.add(keras.layers.Conv1D(filters=128, 
                                  kernel_size=(5), 
                                  activation=activation, 
                                  padding='same', name = 'Conv1D-4'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size = 5, strides = 2, padding='valid', name = 'Maxpool-4'))
    
    # Fifth Convolutional Layer
    model.add(keras.layers.Conv1D(filters=256, 
                                  kernel_size=(5), 
                                  activation=activation, 
                                  padding='same', name = 'Conv1D-5'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling1D(pool_size = 5, strides = 2, padding='valid', name = 'Maxpool-5'))


    # Adding fully connected dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation=activation, name = 'Fully_Connected-1'))
    #model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.Dense(512, activation=activation, name = 'Fully_Connected-2'))
    #model.add(keras.layers.Dropout(0.1))
    
    model.add(keras.layers.Dense(512, activation=activation, name = 'Fully_Connected-3'))
    #model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(512, activation=activation, name = 'Fully_Connected-4'))
    #model.add(keras.layers.Dropout(0.3))
    
    # Output layer with sigmoid function
    model.add(keras.layers.Dense(2, activation="sigmoid", name = 'Sigmoid_Output'))
    
    # setting the optimizer to ADAM and the learning rate to passed in value
    optimizer = keras.optimizers.Adam(learning_rate=learn_rate)
    
    
    # compile the model with Sparse Category Crossentropy, display accuracy metrics
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])

    return model

earlystopping = callbacks.EarlyStopping(monitor ="loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)


# Get all data
training_data = td.train_data()

X_train_cnn = training_data[0]
Y_train_cnn = training_data[1]

X_test_cnn = training_data[2]
Y_test_cnn = training_data[3]
    
X_val_cnn = training_data[4]
Y_val_cnn = training_data[5]

# get planets / non planets
planet = training_data[6]
non_planet = training_data[7]
 


# Reshape and shuffle the data to be used in building the CNN
X_train_cnn = np.expand_dims(X_train_cnn, axis=2)
X_test_cnn = np.expand_dims(X_test_cnn, axis=2)

X_train_cnn, Y_train_cnn = shuffle(X_train_cnn, Y_train_cnn)
    
X_val_cnn = np.expand_dims(X_val_cnn, axis=2)
X_val_cnn, Y_val_cnn = shuffle(X_val_cnn, Y_val_cnn)



# Creation and training of the CNN
modelcnn = create_model(X_train_cnn)
modelcnn.summary()

# Run the CNN without class weights applied
cnn_history = modelcnn.fit(X_train_cnn, 
                           Y_train_cnn, 
                           epochs=50, 
                           batch_size=64, 
                           shuffle=True, 
                           validation_data=(X_val_cnn, Y_val_cnn)
                           )

# get the loss and accuracy metrics for non weighted
loss = cnn_history.history['loss']
acc = cnn_history.history['acc']
epochs = range(1, len(loss)+1)

# plot the training errors for non weighted
plt.title('Training error with epochs \n (CNN)')
plt.plot(epochs, loss, 'bo', label='training loss')
plt.xlabel('epochs')
plt.ylabel('training error')
plt.show()

# plot the accuracy for non weighted
plt.plot(epochs, acc, 'b', label='accuracy')
plt.title('Accuracy of prediction with epochs \n (CNN)')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()



# make predictions on non Weighted CNN
trainPredict_cnn =modelcnn.predict(X_train_cnn)
trainClasses_cnn = np.argmax(trainPredict_cnn,axis=1)
                   
testPredict_cnn = modelcnn.predict(X_test_cnn)
testClasses_cnn = np.argmax(testPredict_cnn,axis=1)


# Create and plot confusion matrix
print('Confusion matrix for the CNN') 
matrix = confusion_matrix(Y_test_cnn,testClasses_cnn)
fig, ax = plt.subplots()
ax.matshow(matrix, cmap=plt.cm.Reds)

for i in range(2):
    for j in range(2):
        c = matrix[j,i]
        ax.text(i, j, str(c), va='center', ha='center')

plt.title('Confusion Matrix for CNN')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# Calculate performance metrics on the test and training data
accuracy_train=sk_met.balanced_accuracy_score(Y_train_cnn,trainClasses_cnn)
accuracy_test= sk_met.balanced_accuracy_score(Y_test_cnn,testClasses_cnn)

precision_train=sk_met.precision_score(Y_train_cnn,trainClasses_cnn)
precision_test=sk_met.precision_score(Y_test_cnn,testClasses_cnn)

recall_train=sk_met.recall_score(Y_train_cnn,trainClasses_cnn)
recall_test=sk_met.recall_score(Y_test_cnn,testClasses_cnn)

# create Dataframe for metrics scores
model_data = {'Metric': ['Balanced Accuracy', 'Precision', 'Recall'],
              'Training Data': [accuracy_train, precision_train, recall_train],
              'Testing Data':[accuracy_test, precision_test, recall_test]}


df_metrics = pd.DataFrame(model_data)
print(df_metrics)

# calculate the F1 scores
f1_score_non_weighted = 2*(recall_test * precision_test) / (recall_test + precision_test)


# create Dataframe for the F1 scores
data_f1_score = {'Type':['CNN'],
                 'F1 Score': [f1_score_non_weighted]}

df_f1_scores = pd.DataFrame(data_f1_score)
print(df_f1_scores)


# Save Created model.
modelcnn.save("./Models/CNN_model.h5")





# ------------------------------  Weights version of model not used 

# =============================================================================
# # creating weights on labels to help with imbalance
# def create_weights_for_labels(Y_train_cnn):
#     neg, pos = np.bincount(np.array(Y_train_cnn, dtype=np.int64))
#     total = neg + pos
#     
#     weight_for_0 = (1 / neg)*(total)/2.0
#     weight_for_1 = (1 / pos)*(total)/2.0
#     
#     class_weight = {0: weight_for_0, 1: weight_for_1}
#     
#     print('Weight for class 0: {:.2f}'.format(weight_for_0))
#     print('Weight for class 1: {:.2f}'.format(weight_for_1))
#     
#     return class_weight
# =============================================================================


# =============================================================================
# # create label weights
# class_weight = training_data[8]
# 
# modelcnn_weighted = create_model(X_train_cnn)
# 
# # # Run the CNN with class weights applied
# cnn_history_weighted = modelcnn_weighted.fit(X_train_cnn, 
#                                               Y_train_cnn, 
#                                              epochs=100, 
#                                               batch_size=64, 
#                                               shuffle=True, 
#                                               class_weight = class_weight, 
#                                               callbacks =[earlystopping]
#                                               )
# 
# # get the loss and accuracy metrics for weighted
# loss_weighted = cnn_history_weighted.history['loss']
# acc_weighted = cnn_history_weighted.history['acc']
# epochs_weighted = range(1, len(loss_weighted)+1)
# 
# # plot the training errors for weighted
# plt.title('Training error with epochs \n (Without class weights)')
# plt.plot(epochs_weighted, loss_weighted, 'bo', label='training loss')
# plt.xlabel('epochs')
# plt.ylabel('training error')
# plt.show()
# 
# # plot the accuracy for weighted
# plt.plot(epochs_weighted, acc_weighted, 'b', label='accuracy')
# plt.title('Accuracy of prediction with epochs \n (Without class weights)')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.show()
# 
# 
# # make predictions on weighted CNN
# trainPredict_cnn_weighted =modelcnn_weighted.predict(X_train_cnn)
# trainClasses_cnn_weighted = np.argmax(trainPredict_cnn_weighted,axis=1)
#                    
# testPredict_cnn_weighted = modelcnn_weighted.predict(X_test_cnn)
# testClasses_cnn_weighted = np.argmax(testPredict_cnn_weighted,axis=1)
# 
# plt.title('Training results (weighted classes)')
# plt.plot(testClasses_cnn_weighted,'*',label='Predicted')
# plt.plot(Y_train_cnn,'o',label='ground truth')
# plt.xlabel('Train data sample index')
# plt.ylabel('Predicted class (0 or 1)')
# plt.legend()
# plt.show()
# 
# 
# plt.title('Performance of the model on testing data (Weighted Classes')
# plt.plot(testClasses_cnn_weighted,'*',label='Predicted')
# plt.plot(Y_test_cnn,'o',label='ground truth')
# plt.xlabel('Test data sample index')
# plt.ylabel('Predicted class (0 or 1)')
# plt.legend()
# plt.show()
# 
# 
# # Create and plot confusion matrix
# print('Confusion matrix for the CNN (Weighted Classes)') 
# matrix = confusion_matrix(Y_test_cnn,testClasses_cnn_weighted)
# fig, ax = plt.subplots()
# ax.matshow(matrix, cmap=plt.cm.Reds)
# 
# for i in range(2):
#      for j in range(2):
#          c = matrix[j,i]
#          ax.text(i, j, str(c), va='center', ha='center')
# 
# 
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()
# 
# 
# # Calculate performance metrics on the test and training data
# accuracy_train_weighted = sk_met.balanced_accuracy_score(Y_train_cnn,trainClasses_cnn_weighted)
# accuracy_test_weighted = sk_met.balanced_accuracy_score(Y_test_cnn,testClasses_cnn_weighted)
# 
# precision_train_weighted = sk_met.precision_score(Y_train_cnn,trainClasses_cnn_weighted)
# precision_test_weighted = sk_met.precision_score(Y_test_cnn,testClasses_cnn_weighted)
# 
# recall_train_weighted = sk_met.recall_score(Y_train_cnn,trainClasses_cnn_weighted)
# recall_test_weighted = sk_met.recall_score(Y_test_cnn,testClasses_cnn_weighted)
# 
# # Create a dataframe for the performance metrics
# model_data_weighted = {'Metric': ['Balanced Accuracy (Weighted)', 'Precision (Weighted)', 'Recall (Weighted)'],
#                'Training Data': [accuracy_train_weighted, precision_train_weighted, recall_train_weighted],
#                'Testing Data':[accuracy_test_weighted, precision_test_weighted, recall_test_weighted]}
# 
# df_metrics_weighted = pd.DataFrame(model_data_weighted)
# 
# # Concatenate all performance metrics into one dataframe
# df_metrics = pd.concat([df_metrics_not_weighted, df_metrics_weighted])
# print(df_metrics)
# 
# # calculate the F1 scores
# f1_score_non_weighted = 2*(recall_test * precision_test) / (recall_test + precision_test)
# f1_score_weighted = 2*(recall_test_weighted * precision_test_weighted) / (recall_test_weighted + precision_test_weighted)
# 
# # create Dataframe for the F1 scores
# data_f1_score = {'Type':['CNN 1 (Not weighted)', 'CNN 2 (Weighted)'],
#                   'F1 Score': [f1_score_non_weighted,f1_score_weighted ]}
# 
# df_f1_scores = pd.DataFrame(data_f1_score)
# print(df_f1_scores)
# =============================================================================


# =============================================================================
# # plot lightcurves with no filtering applied
# def plot_no_filtering(Xtrain, planet, non_planet):
# 
#     # plot what the light curve looks like for an exoplanet star
#     plt.plot(Xtrain[planet], 'r')
#     plt.title('Light intensity vs time (for an exoplanet star)')
#     plt.xlabel('Time index')
#     plt.ylabel('Light intensity')
#     plt.show()
#     
#     # plot what the light curve looks like for an non exoplanet star
#     plt.plot(Xtrain[non_planet], 'b')
#     plt.title('Light intensity vs time (for a non exoplanet star)')
#     plt.xlabel('Time')
#     plt.ylabel('Light intensity')
#     plt.show()
#     
#     
# 
# # Apply a Fourier transformation 
# def apply_fft_and_plot(Xtrain, Xtest, Xval):
#     Xtrain = np.abs(np.fft.rfft(Xtrain, n=len(Xtrain[0]), axis=1))
#     Xtest = np.abs(np.fft.rfft(Xtest, n=len(Xtest[0]), axis=1))
#     Xval = np.abs(np.fft.rfft(Xval, n=len(Xval[0]), axis=1))
# 
#     # Take half of the Fourier spectrum since it is symmetric
#     Xtrain=Xtrain[:,:1+int((len(Xtrain[0])-1)/2)] 
#     Xtest=Xtest[:,:1+int((len(Xtest[0])-1)/2)]
#     Xval=Xval[:,:1+int((len(Xval[0])-1)/2)]
#     
#     
#     plt.plot(Xtrain[planet], 'r')
#     plt.title('After FFT (for an exoplanet star)')
#     plt.xlabel('Frequency')
#     plt.ylabel('Feature value')
#     plt.show()
#     
#     plt.plot(Xtrain[non_planet], 'b')
#     plt.title('After FFT (for a non exoplanet star)')
#     plt.xlabel('Frequency')
#     plt.ylabel('Feature value')
#     plt.show()
#     
#     return Xtrain, Xtest, Xval
# 
# # normalize the data
# def apply_normalization_and_plot(Xtrain, Xtest, Xval):
#     
#     Xtrain = normalize(Xtrain)
#     Xtest = normalize(Xtest)
#     Xval = normalize(Xval)
#     
#     
#     plt.plot(Xtrain[planet], 'r')
#     plt.title('After FFT,Normalization (for an exoplanet star)')
#     plt.xlabel('Frequency')
#     plt.ylabel('Feature value')
#     plt.show()
#     
#     plt.plot(Xtrain[non_planet], 'b')
#     plt.title('After FFT,Normalization (for a non exoplanet star)')
#     plt.xlabel('Frequency')
#     plt.ylabel('Feature value')
#     plt.show()
#     
#     return Xtrain, Xtest, Xval
# 
# 
# # apply gaussian filter
# def apply_gaussin_and_plot(Xtrain, Xtest, Xval):
#     
#     Xtrain = ndimage.filters.gaussian_filter(Xtrain, sigma=50)
#     Xtest = ndimage.filters.gaussian_filter(Xtest, sigma=50)
#     Xval = ndimage.filters.gaussian_filter(Xval, sigma=50)
#     
#     plt.plot(Xtrain[planet], 'r')
#     plt.title('After FFT, Normalization \n and Gaussian filtering (for an exoplanet star)')
#     plt.xlabel('Frequency')
#     plt.ylabel('Feature value')
#     plt.show()
#     
#     plt.plot(Xtrain[non_planet], 'b')
#     plt.title('After FFT, Normalization \n and Gaussian filtering (for a non exoplanet star)')
#     plt.xlabel('Frequency')
#     plt.ylabel('Feature value')
#     plt.show()
#       
#     return Xtrain, Xtest, Xval
# 
# 
# # apply min max scaling 
# def apply_scaler_and_plot(Xtrain, Xtest, Xval):
# 
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     Xtrain = scaler.fit_transform(Xtrain)
#     Xtest = scaler.fit_transform(Xtest)
#     Xval = scaler.fit_transform(Xval)
#     
#     plt.plot(Xtrain[planet], 'r')
#     plt.title('After FFT, Normalization, Gaussian filtering \n and scaling (for an exoplanet star)')
#     plt.xlabel('Frequency')
#     plt.ylabel('Feature value')
#     plt.show()
#     
#     plt.plot(Xtrain[non_planet], 'b')
#     plt.title('After FFT, Normalization, Gaussian filtering \n and scaling (for a non exoplanet star)')
#     plt.xlabel('Frequency')
#     plt.ylabel('Feature value')
#     plt.show()
#     
#     return Xtrain, Xtest, Xval
# =============================================================================



# =============================================================================
# # plot basic data
# plot_no_filtering(X_train_cnn, planet, non_planet)
# 
# # apply fft filtering & plot 
# fft_data = apply_fft_and_plot(X_train_cnn, X_test_cnn, X_val_cnn)
# X_train_cnn = fft_data[0].copy()
# X_test_cnn = fft_data[1].copy()
# X_val_cnn = fft_data[2].copy()
# 
# # apply normalization & plot 
# norm_data = apply_normalization_and_plot(X_train_cnn, X_test_cnn, X_val_cnn)
# X_train_cnn = norm_data[0].copy()
# X_test_cnn = norm_data[1].copy()
# X_val_cnn = norm_data[2].copy()
# 
# # apply gaussian filter & plot 
# gaussian_data = apply_gaussin_and_plot(X_train_cnn, X_test_cnn, X_val_cnn)
# X_train_cnn = gaussian_data[0].copy()
# X_test_cnn = gaussian_data[1].copy()
# X_val_cnn = gaussian_data[2].copy()
# 
# # apply scaler & plot 
# scaler_data = apply_scaler_and_plot(X_train_cnn, X_test_cnn, X_val_cnn)
# X_train_cnn = scaler_data[0].copy()
# X_test_cnn = scaler_data[1].copy()
# X_val_cnn = scaler_data[2].copy()
# =============================================================================
