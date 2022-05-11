# -*- coding: utf-8 -*-
"""

@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import Create_training_data as td
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics as sk_met
from sklearn.preprocessing import normalize



"""
    File 7 
    
    This file is used to run the saved hyper model with accuracy, precision, recall 
    recorded and F1 score calculated and exported to csv. The training accuracy and 
    training loss are plotted for each epoch of the model.
    
    The metrics from the CNN and Capsule Network are imported with McNemars test
    applied to the CNN - CapsNet, CNN-Hypermodel, CapsNet - Hypermodel.
    
"""

# Create Random Seed
np.random.seed(1)


# Get all data
training_data = td.train_data()

X_train = training_data[0]
Y_train = training_data[1]

X_test = training_data[2]
Y_test = training_data[3]
    
X_val = training_data[4]
Y_val = training_data[5]

# get planets / non planets
planet = training_data[6]
non_planet = training_data[7]

# create label weights
class_weight = training_data[8]

# apply gaussian filter
X_train = ndimage.filters.gaussian_filter(X_train, sigma=50)
X_test = ndimage.filters.gaussian_filter(X_test, sigma=50)
X_val = ndimage.filters.gaussian_filter(X_val, sigma=50)

# apply min max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_ = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_val = scaler.fit_transform(X_val)

# shuffle the data set
X_train, Y_train = shuffle(X_train, Y_train)
X_test, Y_test = shuffle(X_test, Y_test)
X_val, Y_val = shuffle(X_val, Y_val)


# Reshape  to be used in testing the CNN
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
X_val = np.expand_dims(X_val, axis=2)


# plot the results for training error and prediction accuracy
def plot_epochs(epochs, loss, acc, name):

    # plot the training errors for weighted
    plt.title('Training error with epochs \n ' + name)
    plt.plot(epochs, loss, 'bo', label='training loss')
    plt.xlabel('epochs')
    plt.ylabel('training error')
    plt.show()
    
    # plot the accuracy for weighted
    plt.plot(epochs, acc, 'b', label='accuracy')
    plt.title('Accuracy of prediction with epochs \n' +name)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
    

# plot the confusion matrix for the tested model
def create_confusion_matrix(Y_test, Classes, name):
    # Create and plot confusion matrix
    matrix = confusion_matrix(Y_test, Classes)
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Reds)
    
    for i in range(2):
        for j in range(2):
            c = matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    
    plt.title('Confusion matrix for ' + name)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    # plot and calculate roc-auc
    fpr, tpr, _ = sk_met.roc_curve(Y_test,  Classes)
    auc = sk_met.roc_auc_score(Y_test, Classes)
    auc1 = f"{auc:.3f}"
    
    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc1))
    plt.title('ROC-AUC for ' + name)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
    
   

# retriece the saved hypermodel created, train on 50 epochs   
def get_hypermodel_cnn(X_train, Y_train, X_val, Y_val):

    # Load models
    hypermodel_cnn = keras.models.load_model("./Models/CNN_Hypermodel.h5")
    hypermodel_cnn.summary()
    stop_early = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    
    hypermodel_cnn_history = hypermodel_cnn.fit(X_train, 
                            Y_train, 
                            epochs = 50, 
                            batch_size=64,
                            validation_data=(X_val, Y_val),
                            #class_weight = class_weight,
                            callbacks=[stop_early])
    
    # get the loss and accuracy metrics for weighted
    hyp_cnn_loss = hypermodel_cnn_history.history['loss']
    hyp_cnn_acc = hypermodel_cnn_history.history['accuracy']
    hyp_cnn_epochs = range(1, len(hyp_cnn_loss)+1)
    
    return hypermodel_cnn, hyp_cnn_loss, hyp_cnn_acc, hyp_cnn_epochs



# run the hypermodel and record predictions on test and train data
def run_hypermodel():
    
    hypermodel = get_hypermodel_cnn(X_train, Y_train, X_val, Y_val)
    
    hypermodel_cnn = hypermodel[0]
    hyp_cnn_loss = hypermodel[1]
    hyp_cnn_acc = hypermodel[2]
    hyp_cnn_epochs = hypermodel[3]
    
    plot_epochs(hyp_cnn_epochs, hyp_cnn_loss, hyp_cnn_acc, 'Hypermodel CNN')    
    
    # make predictions on trainging and test data
    hyp_cnn_trainPredict =hypermodel_cnn.predict(X_train)
    hyp_cnn_trainClasses = [1 if i >=.5 else 0 for i in hyp_cnn_trainPredict]
    
    hyp_cnn_predict = hypermodel_cnn.predict(X_test)
    hcnn_testClasses_cnn = [1 if i >=.5 else 0 for i in hyp_cnn_predict]
    
    create_confusion_matrix(Y_test, hcnn_testClasses_cnn, 'Hypermodel CNN')
    
    return hyp_cnn_trainClasses, hcnn_testClasses_cnn




# compare the models recording, accuracy, precision, recall and F1 score
def model_data():
    
    # run models
    hcnn_data = run_hypermodel()

    
    # get data
    hyp_cnn_trainClasses = hcnn_data[0]
    hcnn_testClasses_cnn = hcnn_data[1]
    
    # Calculate performance metrics on training data
    hcnn_train_accuracy_test= sk_met.accuracy_score(Y_train,hyp_cnn_trainClasses)
    hcnn_train_precision_test=sk_met.precision_score(Y_train,hyp_cnn_trainClasses)
    hcnn_train_recall_test=sk_met.recall_score(Y_train,hyp_cnn_trainClasses)
    
    # Calculate performance metrics on the test data
    hcnn_accuracy_test= sk_met.accuracy_score(Y_test,hcnn_testClasses_cnn)
    hcnn_precision_test=sk_met.precision_score(Y_test,hcnn_testClasses_cnn)
    hcnn_recall_test=sk_met.recall_score(Y_test,hcnn_testClasses_cnn)
    


    # Create a dataframe for the performance metrics
    model_data = {'Metric': ['Accuracy', 'Precision', 'Recall'],
                  'Trainging Data':[hcnn_train_accuracy_test, hcnn_train_precision_test, hcnn_train_recall_test],
                  'Test Data':[hcnn_accuracy_test, hcnn_precision_test, hcnn_recall_test]
                  }


    df_metrics = pd.DataFrame(model_data)
    print(df_metrics)
    #df_metrics.to_excel('../results/tuned_cnn_metrics.xlsx', index=False)
    
    # calculate the F1 scores
    hcnn_f1_score = 2*(hcnn_recall_test * hcnn_precision_test) / (hcnn_recall_test + hcnn_precision_test)

        
    # create Dataframe for the F1 scores
    data_f1_score = {'Type':['F1 Score'],
                     'CNN Hypermodel': [hcnn_f1_score ]}
    
    df_f1_scores = pd.DataFrame(data_f1_score)
    print(df_f1_scores)
    #df_f1_scores.to_excel('../results/tuned_cnn_f1_score.xlsx', index=False)
    
    #classes_output = pd.DataFrame(hcnn_testClasses_cnn)
    #classes_output.to_excel('../results/tuned_cnn_predictions.xlsx', index=False)
    #actual_classes_output = pd.DataFrame(Y_test)
    #actual_classes_output.to_excel('../results/tuned_cnn_actual.xlsx', index=False)
    
    
# calling main function.
if __name__ == "__main__":
    model_data()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    