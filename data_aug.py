#!/usr/bin/env python
# coding: utf-8

from keras import models
from keras import layers
from keras.utils import np_utils, generic_utils
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from sklearn.model_selection import KFold
from keras.optimizers import adam
import pickle
import numpy as np


### Get data from the input file ###

infile = open('/headnode2/mngu6638/mproject/train_data','rb')
Images = pickle.load(infile)
Labels = pickle.load(infile)
infile.close()
print('Finish importing data')
### Data ###

# Convert labels into binary #
binaryLabels = np_utils.to_categorical(Labels)

# Training data #
xtrain = Images
ytrain = binaryLabels
print('Checking the training data')
print(xtrain.shape)
print(ytrain.shape)


### Create a datagen for transformation
datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
g = datagen.flow(xtrain, ytrain, batch_size=1)
print('Finish defining ImageDataGenerator')

# Create variables to hold the transformed data
# One for X, one for Y
trDataX = []
trDataY = []


# Run the generator multiples times
N = len(xtrain) * 20
for i in range(N):
  xBatch, yBatch = next(g)
  trDataX.append(xBatch)
  trDataY.append(yBatch)


# Convert the list to np.array
# np.concatenate will merge the data along an axis
trDataX = np.concatenate(trDataX, axis=0)
trDataY = np.concatenate(trDataY, axis=0)


# Check the dimension
print('Finish data augmentation process')
print('Check the dimension')
print(trDataX.shape)
print(trDataY.shape)


# Use the data for model training

# Save the data

with open ('/headnode2/mngu6638/mproject/trData', 'wb') as f:
    pickle.dump(trDataX, f)
    pickle.dump(trDataY, f)
