# -*- coding: utf-8 -*-
"""

@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland

"""

import numpy as np
import pandas as pd
from keras import layers, models
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import tensorflow as tf
from keras import callbacks
from sklearn import metrics as sk_met
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix




"""
    File 10 
    
    This file bulds the Capsule Network using the custom layers created in capsulelayers.py.
    The data is shuffled and reshaped to be able to be passed into the network by adding an 
    additional dimesion. The first layer two layers of the network are convolutional layers 
    with a max pooling latyer inbetween. The next layer is the primary capsule where the another
    convolutional layer is used in conjuction with rehsaping and the custom squash function.
    
    The fourth layer is where the capsule network applies it's main work by creating pairs of 
    calculated outputs based on the number of classes (2) and the number of dimensions specified
    each capsule giving a prediction of the outcome. These are then routed through the agreement
    function outputting a predicted value for each class.
    
    A decoder network is constructed 
    
"""

# Create Random Seed
np.random.seed(1)

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


# Create the capsule network
def CapsNet(input_shape, n_class, routings):

    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv1D layer
    conv1 = layers.Conv1D(filters=112, 
                          kernel_size=5, 
                          padding='valid', 
                          activation='relu', 
                          name='conv1')(x)
    
    pooling1 = layers.MaxPooling1D(pool_size = 5, 
                                   strides = 2)(conv1)
    
    # Layer 2: Just a second conventional Conv1D layer
    conv2 = layers.Conv1D(filters=256, 
                          kernel_size=5,  
                          padding='same', 
                          activation='relu', 
                          name='conv2')(pooling1)
    


    
    # Layer 3: Conv1D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv2, 
                             dim_capsule=2, 
                             n_channels=20, 
                             kernel_size=5, 
                             strides=2, 
                             padding='same')
    
    # Layer 4: Capsule layer. Routing algorithm works here.
    secondary_caps = CapsuleLayer(num_capsule=n_class, 
                                  dim_capsule=2, 
                                  num_routing=routings,
                                  name='secondary_caps')(primarycaps)
    
    # Layer 5: An auxiliary layer to replace each capsule with its length to match the true label's shape.
    out_caps = Length(name='capsnet')(secondary_caps) 

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    # applying mask
    masked = Mask()(secondary_caps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    # add fully connected layer to the decoder
    decoder.add(layers.Dense(512, activation='relu', input_dim=2*n_class))
    
    # sigmoid output layer
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked)]) #masked_by_y
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 2))
    noised_secondary_caps = layers.Add()([secondary_caps, noise])
    masked_noised_y = Mask()([noised_secondary_caps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    
    
    return train_model, eval_model, manipulate_model


# train the model and plot training results
def train(model, X_train, y_train, X_test, y_test, num):
   
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
                  loss=tf.keras.losses.BinaryCrossentropy(
                                                        from_logits=False,
                                                        label_smoothing=0.0,
                                                        axis=-1,
                                                        reduction="auto",
                                                        name="binary_crossentropy",),
                  #loss_weights=[1.86, 0.68],
                  metrics={'capsnet': 'accuracy'})
    
    # print summary of model
    model.summary()
    
    earlystopping = callbacks.EarlyStopping(monitor ="loss", 
                                        mode ="min", patience = 2, 
                                        restore_best_weights = True)
    
    # Training model:
    hist = model.fit([X_train, y_train], 
                     [y_train, X_train], 
                     batch_size=64, 
                     epochs=20,
                     shuffle=True,
                     callbacks =[earlystopping]
                     )
    
    # get the loss and accuracy metrics for weighted
    capsnet_loss = hist.history['loss']
    capsnet_acc = hist.history['capsnet_accuracy']
    capsnet_epochs = range(1, len(capsnet_loss)+1)
    
    plot_epochs(capsnet_epochs, capsnet_loss, capsnet_acc, f'CapsNet(routings: {num})')
    
    # Save Created model.
    #model.save("./Models/CapsNet_model.h5")
    
    return model


# test the model
def test(model, X_test, y_test):
    # make predictions
    y_pred, x_recon = model.predict(X_test)
    
    return y_pred


def manipulate_latent(model, X_train, y_train, X_test, y_test):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    index = np.argmax(y_test, 1)
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = X_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)


def get_data():
    # Import the data
    df1 = pd.read_csv('D:/College/year_4/semester_2/Software_project/Discovery2/data/Output/train_data.csv')
    df2 = pd.read_csv('D:/College/year_4/semester_2/Software_project/Discovery2/data/Output/test_data.csv')
    df3 = pd.read_csv('D:/College/year_4/semester_2/Software_project/Discovery2/data/Output/val_data.csv')
    

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
    
    # print shape of training X and Y 
    print('Shape of Xtrain:', np.shape(Xtrain), '\nShape of ytrain:', np.shape(ytrain))
    
    return ytrain, Xtrain, ytest, Xtest, yval, Xval


# load the data
def load_data():
    # Get all data
    data = get_data()
    ytrain = data[0].copy()
    Xtrain = data[1].copy()
    ytest = data[2].copy()
    Xtest = data[3].copy()

    # Training Data copies
    y_train = ytrain.copy()
    x_train = Xtrain.copy()
    
    # Test data copies
    y_test = ytest.copy()
    x_test = Xtest.copy()
    
    # apply gaussian filter
    x_train = ndimage.filters.gaussian_filter(x_train, sigma=50)
    x_test = ndimage.filters.gaussian_filter(x_test, sigma=50)
    
    # apply mix max scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    
    # pre shuffle the data set
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    
    # changing y train to categorical
    y_train = to_categorical(y_train)

    # reshaping for input 
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


    return x_train, y_train, x_test, y_test


# build and train the model
def capsule_net():
    """
    # epochs = 50
    # batch_size = 64
    # lr = 0.001
    # lr_decay = 0
    # lam_recon = 0.392
    # routings = 3
    # shift_fraction = 0.1
    """
    # load data
    X_train,y_train,X_test,y_test = load_data()
    
    # define model with 1 routings
    model, eval_model, manipulate_model = CapsNet(input_shape=X_train[1,:,:].shape,
                                                  n_class=2,
                                                  routings=1)
    
    # define model with 2 routings
    model2, eval_model2, manipulate_model2 = CapsNet(input_shape=X_train[1,:,:].shape,
                                                  n_class=2,
                                                  routings=2)
    
    # define model with 3 routings
    model3, eval_model3, manipulate_model3 = CapsNet(input_shape=X_train[1,:,:].shape,
                                                  n_class=2,
                                                  routings=3)
    
    # define model with 4 routings
    model4, eval_model4, manipulate_model4 = CapsNet(input_shape=X_train[1,:,:].shape,
                                                  n_class=2,
                                                  routings=4)
    
    # define model with 5 routings
    model5, eval_model5, manipulate_model5 = CapsNet(input_shape=X_train[1,:,:].shape,
                                                  n_class=2,
                                                  routings=5)
    
    # train models
    train(model, X_train, y_train, X_test, y_test, '1')
    train(model2, X_train, y_train, X_test, y_test, '2')
    train(model3, X_train, y_train, X_test, y_test, '3')
    train(model4, X_train, y_train, X_test, y_test, '4')
    train(model5, X_train, y_train, X_test, y_test, '5')

    return eval_model, eval_model2, eval_model3, eval_model4, eval_model5


# test the returned model, print and save resutls
def evaluate_model(eval_model, num):
    
    # load data
    X_train,y_train,X_test,y_test = load_data()
    
    # make predictions on trainging and test data
    capsnet_trainPredict, x_recon = eval_model.predict(X_train)
    capsnet_trainClasses = np.argmax(capsnet_trainPredict,1)

    pred = test(eval_model, X_test, y_test)
    testClasses_capsnet = np.argmax(pred,1)
    
    # save predictions versus actuals
    #classes_output = pd.DataFrame(testClasses_capsnet)
    #classes_output.to_excel(f'../results/capsent_predictions{num}.xlsx', index=False)
    #actual_classes_output = pd.DataFrame(y_test)
    #actual_classes_output.to_excel(f'../results/capsent_actual{num}.xlsx', index=False)

    # plot the confusion matrix
    create_confusion_matrix(y_test, testClasses_capsnet, f'CapsNet {num}')
    
    # retreive original copy of training data
    data = get_data()
    Y_train = data[0].copy()
    
    # Calculate performance metrics on training data
    capsnet_train_accuracy_test= sk_met.accuracy_score(Y_train,capsnet_trainClasses)
    capsnet_train_precision_test=sk_met.precision_score(Y_train,capsnet_trainClasses)
    capsnet_train_recall_test=sk_met.recall_score(Y_train,capsnet_trainClasses)
    
    # Calculate performance metrics on the test data
    capsnet_accuracy_test= sk_met.accuracy_score(y_test, testClasses_capsnet)
    capsnet_precision_test=sk_met.precision_score(y_test, testClasses_capsnet)
    capsnet_recall_test=sk_met.recall_score(y_test, testClasses_capsnet)
    
    # Create a dataframe for the performance metrics
    model_data = {'Metric': ['Accuracy', 'Precision', 'Recall'],
                  f'Capsnet(routings:{num}) Trainging':[capsnet_train_accuracy_test, capsnet_train_precision_test, capsnet_train_recall_test],
                  f'Capsnet(routings:{num}) Testing':[capsnet_accuracy_test, capsnet_precision_test, capsnet_recall_test]
                  }
    
    # convert to a dataframe and save
    df_metrics = pd.DataFrame(model_data)
    print(df_metrics)
    #df_metrics.to_excel(f'../results/capsent_metrics{num}.xlsx', index=False)
    
    # calculate the F1 scores
    capsnet_f1_score = 2*(capsnet_recall_test * capsnet_precision_test) / (capsnet_recall_test + capsnet_precision_test)
    
    # create Dataframe for the F1 scores
    data_f1_score = {'Type':['F1 Score'],
                     f'CapsNet(routings:{num})': [capsnet_f1_score ]}
    
    # convert to a dataframe and save
    df_f1_score = pd.DataFrame(data_f1_score)
    print(df_f1_score)
    #df_f1_score.to_excel(f'../results/capsent_f1_score{num}.xlsx', index=False)
    

# calling main function.
if __name__ == "__main__":
    
    # return the trained model
    model = capsule_net()
    
    model1 = model[0]
    model2 = model[1]
    model3 = model[2]
    model4 = model[3]
    model5 = model[4]
    
    # evaluate the returned model
    evaluate_model(model1, '1')
    evaluate_model(model2, '2')
    evaluate_model(model3, '3')
    evaluate_model(model4, '4')
    evaluate_model(model5, '5')
    
    

    