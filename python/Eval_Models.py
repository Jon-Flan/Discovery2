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
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics as sk_met
from sklearn.preprocessing import normalize

"""
    File 10 
    
    This file is used to ......
    
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



# Reshape and shuffle the data to be used in building the CNN
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

X_train, Y_train = shuffle(X_train, Y_train)
    
X_val = np.expand_dims(X_val, axis=2)
X_val, Y_val = shuffle(X_val, Y_val)


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
    

# retriece the saved hypermodel created, train on 50 epochs   
def get_hypermodel_cnn(X_train, Y_train, X_val, Y_val):

    # Load models
    hypermodel_cnn = keras.models.load_model("./Models/CNN_Hypermodel.h5")
    hypermodel_cnn.summary()
    #stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    hypermodel_cnn_history = hypermodel_cnn.fit(X_train, 
                            Y_train, 
                            epochs = 50, 
                            batch_size=64)
    
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
    hyp_cnn_trainClasses = np.argmax(hyp_cnn_trainPredict,axis=1)
    
    hyp_cnn_predict = hypermodel_cnn.predict(X_test)
    hcnn_testClasses_cnn = np.argmax(hyp_cnn_predict,axis=1)
    
    create_confusion_matrix(Y_test, hcnn_testClasses_cnn, 'Hypermodel CNN')
    
    return hyp_cnn_trainClasses, hcnn_testClasses_cnn


# run the manually made cnn and train on 50 epochs 
def get_cnn(X_train, Y_train, X_val, Y_val):
    # Load models
    cnn_model = keras.models.load_model("./Models/CNN_model.h5")
    cnn_model.summary()
    #stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    cnn_history = cnn_model.fit(X_train, 
                            Y_train, 
                            epochs = 50, 
                            batch_size=64)
    
    # get the loss and accuracy metrics for weighted
    cnn_loss = cnn_history.history['loss']
    cnn_acc = cnn_history.history['acc']
    cnn_epochs = range(1, len(cnn_loss)+1)
    
    return cnn_model, cnn_loss, cnn_acc, cnn_epochs


# run the hypermodel and record predictions on test and train data
def run_cnn_model():
    
    cnn_model = get_cnn(X_train, Y_train, X_val, Y_val)
    
    cnn = cnn_model[0]
    cnn_loss = cnn_model[1]
    cnn_acc = cnn_model[2]
    cnn_epochs = cnn_model[3]
    
    plot_epochs(cnn_epochs, cnn_loss, cnn_acc, 'CNN')    
    
    # make predictions on trainging and test data
    cnn_trainPredict = cnn.predict(X_train)
    cnn_trainClasses = np.argmax(cnn_trainPredict,axis=1)
    
    cnn_predict = cnn.predict(X_test)
    testClasses_cnn = np.argmax(cnn_predict,axis=1)
    
    create_confusion_matrix(Y_test, testClasses_cnn, 'CNN')
    
    return cnn_trainClasses, testClasses_cnn


# run the created capsule network and train on 50 epochs 
def get_capsnet(X_train, Y_train):
    # Load models
    capsnet_model = keras.models.load_model("./Models/CapsNet_model.h5")
    capsnet_model.summary()
    
    # changing y train to categorical
    Y_train = to_categorical(Y_train)
    
    # reshaping for input 
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    
    capsnet_hist = capsnet_model.fit([X_train, Y_train], [Y_train, X_train], batch_size=64, epochs=50,
               shuffle=True)
    
    # get the loss and accuracy metrics for weighted
    capsnet_loss = capsnet_hist.history['loss']
    capsnet_acc = capsnet_hist.history['capsnet_accuracy']
    capsnet_epochs = range(1, len(capsnet_loss)+1)
    
    return capsnet_model, capsnet_loss, capsnet_acc, capsnet_epochs

# run the capsule network and record predictions on test and train data
def run_capsnet_model():
    # Get all data
    training_data = td.train_data()
    
    X_train = training_data[0]
    Y_train = training_data[1]
    
    X_test = training_data[2]
    Y_test = training_data[3]
    
    capsnet_model = get_capsnet(X_train, Y_train)
    
    capsnet = capsnet_model[0]
    capsnet_loss = capsnet_model[1]
    capsnet_acc = capsnet_model[2]
    capsnet_epochs = capsnet_model[3]
    
    plot_epochs(capsnet_epochs, capsnet_loss, capsnet_acc, 'CapsNet') 
    
    # reshaping for input 
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # make predictions on trainging and test data
    capsnet_trainPredict = capsnet.predict(X_train)
    capsnet_trainClasses = np.argmax(capsnet_trainPredict,axis=1)
    
    capsnet_predict = capsnet.predict(X_test)
    testClasses_capsnet = np.argmax(capsnet_predict,axis=1)
    
    create_confusion_matrix(Y_test, testClasses_capsnet, 'CapsNet')
    
    return capsnet_trainClasses, testClasses_capsnet


# compare the models recording, accuracy, precision, recall and F1 score
def compare_models():
    
    # run models
    hcnn_data = run_hypermodel()
    cnn_data = run_cnn_model()
    capsnet_data = run_capsnet_model()
    
    # get data
    hyp_cnn_trainClasses = hcnn_data[0]
    hcnn_testClasses_cnn = hcnn_data[1]
    
    cnn_trainClasses = cnn_data[0]
    testClasses_cnn = cnn_data[1]
    
    capsnet_trainClasses = capsnet_data[0]
    testClasses_capsnet = capsnet_data[1]
    
    # Calculate performance metrics on training data
    hcnn_train_accuracy_test= sk_met.balanced_accuracy_score(Y_train,hyp_cnn_trainClasses)
    hcnn_train_precision_test=sk_met.precision_score(Y_train,hyp_cnn_trainClasses)
    hcnn_train_recall_test=sk_met.recall_score(Y_train,hyp_cnn_trainClasses)
    
    cnn_train_accuracy_test= sk_met.balanced_accuracy_score(Y_train,cnn_trainClasses)
    cnn_train_precision_test=sk_met.precision_score(Y_train,cnn_trainClasses)
    cnn_train_recall_test=sk_met.recall_score(Y_train,cnn_trainClasses)
    
    capsnet_train_accuracy_test= sk_met.balanced_accuracy_score(Y_train,capsnet_trainClasses)
    capsnet_train_precision_test=sk_met.precision_score(Y_train,capsnet_trainClasses)
    capsnet_train_recall_test=sk_met.recall_score(Y_train,capsnet_trainClasses)
    
    
    # Calculate performance metrics on the test data
    hcnn_accuracy_test= sk_met.balanced_accuracy_score(Y_test,hcnn_testClasses_cnn)
    hcnn_precision_test=sk_met.precision_score(Y_test,hcnn_testClasses_cnn)
    hcnn_recall_test=sk_met.recall_score(Y_test,hcnn_testClasses_cnn)
    
    cnn_accuracy_test= sk_met.balanced_accuracy_score(Y_test, testClasses_cnn)
    cnn_precision_test=sk_met.precision_score(Y_test, testClasses_cnn)
    cnn_recall_test=sk_met.recall_score(Y_test, testClasses_cnn)
    
    capsnet_accuracy_test= sk_met.balanced_accuracy_score(Y_test, testClasses_capsnet)
    capsnet_precision_test=sk_met.precision_score(Y_test, testClasses_capsnet)
    capsnet_recall_test=sk_met.recall_score(Y_test, testClasses_capsnet)


    # Create a dataframe for the performance metrics
    model_data = {'Metric': ['Balanced Accuracy', 'Precision', 'Recall'],
                  'Hypermodel Trainging Data':[hcnn_train_accuracy_test, hcnn_train_precision_test, hcnn_train_recall_test],
                  'Hypermodel Testing Data':[hcnn_accuracy_test, hcnn_precision_test, hcnn_recall_test], 
                  'CNN Trainging Data':[cnn_train_accuracy_test, cnn_train_precision_test, cnn_train_recall_test],
                  'CNN Testing Data':[cnn_accuracy_test, cnn_precision_test, cnn_recall_test],
                  'Capsnet Trainging Data':[capsnet_train_accuracy_test, capsnet_train_precision_test, capsnet_train_recall_test],
                  'Capsnet Testing Data':[capsnet_accuracy_test, capsnet_precision_test, capsnet_recall_test]
                  }


    df_metrics = pd.DataFrame(model_data)
    print(df_metrics)
    
    # calculate the F1 scores
    hcnn_f1_score = 2*(hcnn_recall_test * hcnn_precision_test) / (hcnn_recall_test + hcnn_precision_test)
    cnn_f1_score = 2*(cnn_recall_test * cnn_precision_test) / (cnn_recall_test + cnn_precision_test)
    capsnet_f1_score = 2*(capsnet_recall_test * capsnet_precision_test) / (capsnet_recall_test + capsnet_precision_test)
    
    
    # create Dataframe for the F1 scores
    data_f1_score = {'Type':['F1 Score'],
                     'CNN Hypermodel': [hcnn_f1_score ],
                     'CNN Manual': [cnn_f1_score ],
                     'CapsNet': [capsnet_f1_score ]}
    
    df_f1_scores = pd.DataFrame(data_f1_score)
    print(df_f1_scores)
    
    
# calling main function.
if __name__ == "__main__":
    compare_models()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    