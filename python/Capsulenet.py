# -*- coding: utf-8 -*-
"""
@author: Jonathan Flanagan
Student Number: x18143890

Software Project for BSHCEDA4 National College of Ireland
"""

import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import tensorflow as tf


"""
    File 9 
    
    This file is used to ......
    
"""

# Create Random Seed
np.random.seed(1)

def CapsNet(input_shape, n_class, routings):

    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv1D layer
    conv1 = layers.Conv1D(filters=112, kernel_size=5, strides=4, padding='same', activation='relu', name='conv1')(x)
    
    pooling = layers.MaxPooling1D(pool_size = 4, strides = 4)(conv1)
    
    # 
    conv2 = layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu', name='conv2')(pooling)
    
    # Layer 2: Conv1D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv2, dim_capsule=2, n_channels=20, kernel_size=3, strides=2, padding='same')
    
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=10, num_routing=routings,
                             name='digitcaps')(primarycaps)
    
    # Layer 4: An auxiliary layer to replace each capsule with its length to match the true label's shape.
    out_caps = Length(name='capsnet')(digitcaps) 

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
#    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid', input_dim=10*n_class))
    decoder.add(layers.Dense(32, activation='relu', input_dim=10*n_class))
#    decoder.add(layers.Dense(16, activation='relu'))
    decoder.add(layers.Dropout(0.20))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked)]) #masked_by_y
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 10))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):

    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.9 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.05))

    return K.mean(K.sum(L, 1))


def train(model, X_train, y_train, X_test, y_test):
   
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
                  loss=[margin_loss, 'categorical_crossentropy'],
                  loss_weights=[1, 0.392],
                  metrics={'capsnet': 'accuracy'})

    # Training model:
    hist = model.fit([X_train, y_train], [y_train, X_train], batch_size=64, epochs=50,
               shuffle=True)
    
    # Save Created model.
    model.save("./Models/CapsNet_model.h5")
    
    return model


def test(model, X_test, y_test):
    y_pred, x_recon = model.predict(X_test, batch_size=20)
    print('-'*30 + 'Begin: test' + '-'*30)
    #print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 0))/y_test.shape[0])
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
    
    # changing y train to categorical
    y_train = to_categorical(y_train)
    
    # reshaping for input 
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test


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
    
    
    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=X_train[1,:,:].shape,
                                                  n_class=2,
                                                  routings=3)
    # output model summary
    model.summary()

    # train and test
    train(model, X_train, y_train, X_test, y_test)
    pred = test(eval_model, X_test, y_test)
    
    return model
    
capsule_net()   
    
    