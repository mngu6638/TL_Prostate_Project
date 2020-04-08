#!/usr/bin/env python
# coding: utf-8

# Import lib

from keras import models
from keras import layers
from keras.utils import np_utils, generic_utils
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from keras.optimizers import adam
from keras.callbacks import CSVLogger
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np

### Get data from the input file ###

infile = open('/headnode2/mngu6638/mproject/trData','rb')
Images = pickle.load(infile)
Labels = pickle.load(infile)
infile.close()
print('All data is loaded')


### Data ###

# Training data #
xtrain = Images
ytrain = Labels
print(xtrain.shape)
print(ytrain.shape)


### CNN model and training ###

base_model = VGG19(weights=None, include_top=False, input_shape=(224,224,3))
base_model.load_weights('/headnode2/mngu6638/mproject/vgg19.h5')
# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)
acc_per_fold = []       # Define per-fold score containers
loss_per_fold = []      # Define per-fold score containers

fold_no = 1
for train, val in kfold.split(xtrain, ytrain):
    X_train = xtrain[train]
    X_val = xtrain[val]
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    y_train = ytrain[train]
    y_val = ytrain[val]

    
    # Model architecture #
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))
    for layer in base_model.layers:
        layer.trainable = False
    model.summary()
    
    
    # Compile the model
    opt = adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    print('------------------------------------------------------------------------')
    print(fold_no)
    BS = 64		#Batch-size
    Epochs_no = 10	#Epochs
    CB = CSVLogger('/headnode2/mngu6638/mproject/VGG19_ptmodel_save/VGG19PTCW_Test3_Fold'+str(fold_no)+'_CB.csv',separator = ',', append=False)
    class_weights = {0:1,1:3.2}


    # Train the model 
    history = model.fit(X_train, y_train, epochs=Epochs_no, batch_size=BS,validation_data=(X_val,y_val), callbacks = [CB], class_weight=class_weights, verbose=0)
    
    
    
    # Save model #
    model.save('/headnode2/mngu6638/mproject/VGG19_ptmodel_save/VGG19PTCW_Test3_Fold'+str(fold_no)+'.h5')
    
    # Generate generalization metrics
    scores = model.evaluate(X_val, y_val)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])


    # Increase fold number
    fold_no = fold_no + 1



# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

