#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# johanTraining.py ////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains an image only version of BEST ///////////////////////////////
#==================================================================================

# modules
import numpy
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import copy
import random

# get stuff from modules
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, SeparableConv2D, Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, MaxoutDense
from keras.layers import GRU, LSTM, ConvLSTM2D, Reshape
from keras.regularizers import l1,l2
from keras.utils import np_utils, to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# set up gpu environment
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
k.tensorflow_backend.set_session(tf.Session(config=config))

# user modules
import tools.functions as tools

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

# set options 
savePDF = False
savePNG = True

setTypes = ["train","validation","test"]
sampleTypes = ["QCD","b","W","Z","H","t"]
frameTypes = ["W","Z","H","t"]

#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
# put images in data frames
jetDF = {} # Keys are "mySample_myKey_mySet", i.e. QCD_HiggsFrame_images_train
for mySet in setTypes:
   for mySample in sampleTypes:
      myF = h5py.File("/uscms/home/bonillaj/nobackup/h5samples/"+mySample+"Sample_BESTinputs_"+myType+"_flattened.h5","r")
      for myKey in myF.keys():
         jetDF[mySample+"_"+myKey+"_"+mySet] = myF[myKey][()]
      myF.close()

print("Accessed Jet Images")
print("Made image dataframes")

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Store data and truth
# print("Number of QCD Jet Images: ", len(jetDF['QCD_HiggsFrame_images_train']) )
# print("Number of Higgs Jet Images: ", len(jetImagesDF['HH']) )
# print("Number of W Jet Images: ", len(jetImagesDF['WW']) )
# print("Number of Z Jet Images: ", len(jetImagesDF['ZZ']) )
# print("Number of t Jet Images: ", len(jetImagesDF['tt']) )
# print("Number of b Jet Images: ", len(jetImagesDF['bb']) )

## Order of categories: 0-W, 1-Z, 2-H, 3-t, 4-b, 5-QCD
truthLabelsTrain = numpy.concatenate([numpy.zeros(len(jetDF['W_BES_vars_train']) ), numpy.ones(len(jetImagesDF['Z_BES_vars_train']) ), numpy.full(len(jetImagesDF['H_BES_vars_train']), 2),
                            numpy.full(len(jetImagesDF['t_BES_vars_train']), 3), numpy.full(len(jetImagesDF['b_BES_vars_train']), 4), numpy.full(len(jetImagesDF['QCD_BES_vars_train']), 5)] )
truthLabelsValidation = numpy.concatenate([numpy.zeros(len(jetDF['W_BES_vars_validation']) ), numpy.ones(len(jetImagesDF['Z_BES_vars_validation']) ), numpy.full(len(jetImagesDF['H_BES_vars_validation']), 2),
                            numpy.full(len(jetImagesDF['t_BES_vars_validation']), 3), numpy.full(len(jetImagesDF['b_BES_vars_validation']), 4), numpy.full(len(jetImagesDF['QCD_BES_vars_validation']), 5)] )
truthLabelsTrain = to_categorical(truthLabelsTrain, num_classes = 6)
truthLabelsValidation = to_categorical(truthLabelsValidation, num_classes = 6)

jetHImageTrain = numpy.concatenate([jetDF['W_HiggsFrame_images_train'], jetImagesDF['Z_HiggsFrame_images_train'], jetImagesDF['Z_HiggsFrame_images_train'], jetImagesDF['t_HiggsFrame_images_train'], jetImagesDF['b_HiggsFrame_images_train'], jetImagesDF['QCD_HiggsFrame_images_train'] ])
jetHImageValidation = numpy.concatenate([jetDF['W_HiggsFrame_images_validation'], jetImagesDF['Z_HiggsFrame_images_validation'], jetImagesDF['Z_HiggsFrame_images_validation'], jetImagesDF['t_HiggsFrame_images_validation'], jetImagesDF['b_HiggsFrame_images_validation'], jetImagesDF['QCD_HiggsFrame_images_validation'] ])

jetWImageTrain = numpy.concatenate([jetDF['W_WFrame_images_train'], jetImagesDF['Z_WFrame_images_train'], jetImagesDF['Z_WFrame_images_train'], jetImagesDF['t_WFrame_images_train'], jetImagesDF['b_WFrame_images_train'], jetImagesDF['QCD_WFrame_images_train'] ])
jetWImageValidation = numpy.concatenate([jetDF['W_WFrame_images_validation'], jetImagesDF['Z_WFrame_images_validation'], jetImagesDF['Z_WFrame_images_validation'], jetImagesDF['t_WFrame_images_validation'], jetImagesDF['b_WFrame_images_validation'], jetImagesDF['QCD_WFrame_images_validation'] ])

jetZImageTrain = numpy.concatenate([jetDF['W_ZFrame_images_train'], jetImagesDF['Z_ZFrame_images_train'], jetImagesDF['Z_ZFrame_images_train'], jetImagesDF['t_ZFrame_images_train'], jetImagesDF['b_ZFrame_images_train'], jetImagesDF['QCD_ZFrame_images_train'] ])
jetZImageValidation = numpy.concatenate([jetDF['W_ZFrame_images_validation'], jetImagesDF['Z_ZFrame_images_validation'], jetImagesDF['Z_ZFrame_images_validation'], jetImagesDF['t_ZFrame_images_validation'], jetImagesDF['b_ZFrame_images_validation'], jetImagesDF['QCD_ZFrame_images_validation'] ])

jetTImageTrain = numpy.concatenate([jetDF['W_TopFrame_images_train'], jetImagesDF['Z_TopFrame_images_train'], jetImagesDF['Z_TopFrame_images_train'], jetImagesDF['t_TopFrame_images_train'], jetImagesDF['b_TopFrame_images_train'], jetImagesDF['QCD_TopFrame_images_train'] ])
jetTImageValidation = numpy.concatenate([jetDF['W_TopFrame_images_validation'], jetImagesDF['Z_TopFrame_images_validation'], jetImagesDF['Z_TopFrame_images_validation'], jetImagesDF['t_TopFrame_images_validation'], jetImagesDF['b_TopFrame_images_validation'], jetImagesDF['QCD_TopFrame_images_validation'] ])

jetImagesTrain = numpy.concatenate([jetHImageTrain, jetWImageTrain, jetZImageTrain, jetTImageTrain ])
jetImagesValidation = numpy.concatenate([jetHImageValidation, jetWImageValidation, jetZImageValidation, jetTImageValidation ])
jetImagesTrain = to_categorical(jetImagesTrain, num_classes = 6)
jetImagesValidation = to_categorical(jetImagesValidation, num_classes = 6)


jetBESvarsTrain = numpy.concatenate([jetDF['W_BES_vars_train'], jetImagesDF['Z_BES_vars_train'], jetImagesDF['Z_BES_vars_train'], jetImagesDF['t_BES_vars_train'], jetImagesDF['b_BES_vars_train'], jetImagesDF['QCD_BES_vars_train'] ])
jetBESvarsValidation = numpy.concatenate([jetDF['W_BES_vars_validation'], jetImagesDF['Z_BES_vars_validation'], jetImagesDF['Z_BES_vars_validation'], jetImagesDF['t_BES_vars_validation'], jetImagesDF['b_BES_vars_validation'], jetImagesDF['QCD_BES_vars_validation'] ])

jetBESvarsTrain = to_categorical(jetBESvarsTrain, num_classes = 6)
jetBESvarsValidation = to_categorical(jetBESvarsValidation, num_classes = 6)

print("Stored data and truth information")

#print("Number of QCD jets in training: ", numpy.sum(trainTruth == 0) )
#print("Number of H jets in training: ", numpy.sum(trainTruth == 1) )
#print("Number of W jets in training: ", numpy.sum(trainTruth == 2) )

#print("Number of QCD jets in testing: ", numpy.sum(testTruth == 0) )
#print("Number of H jets in testing: ", numpy.sum(testTruth == 1) )
#print("Number of W jets in testing: ", numpy.sum(testTruth == 2) )


# Define the Neural Network Structure
#print("NN input shape: ", trainData.shape[1], trainData.shape[2], trainData.shape[3] )

# put images and BES variables in data frames
arbitrary_length = 10 #Hopefully this number doesn't matter
nx = 31
ny = 31
ImageShapeHolder = numpy.zeros((arbitrary_length, nx, ny, 1))
BestShapeHolder = 94

BatchSize = 1200

HiggsImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )

HiggsImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageInputs)
HiggsImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = BatchNormalization(momentum = 0.6)(HiggsImageLayer)
HiggsImageLayer = MaxPool2D(pool_size=(2,2) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = BatchNormalization(momentum = 0.6)(HiggsImageLayer)
HiggsImageLayer = MaxPool2D(pool_size=(2,2) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
HiggsImageLayer = BatchNormalization(momentum = 0.6)(HiggsImageLayer)
HiggsImageLayer = MaxPool2D(pool_size=(2,2) )(HiggsImageLayer) 
HiggsImageLayer = Flatten()(HiggsImageLayer)
HiggsImageLayer = Dropout(0.20)(HiggsImageLayer)
#HiggsImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
#HiggsImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
#HiggsImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
#HiggsImageLayer = Dropout(0.10)(HiggsImageLayer)

HiggsImageModel = Model(inputs = HiggsImageInputs, outputs = HiggsImageLayer)

#Top image
TopImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )

TopImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageInputs)
TopImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = BatchNormalization(momentum = 0.6)(TopImageLayer)
TopImageLayer = MaxPool2D(pool_size=(2,2) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = BatchNormalization(momentum = 0.6)(TopImageLayer)
TopImageLayer = MaxPool2D(pool_size=(2,2) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
TopImageLayer = BatchNormalization(momentum = 0.6)(TopImageLayer)
TopImageLayer = MaxPool2D(pool_size=(2,2) )(TopImageLayer)
TopImageLayer = Flatten()(TopImageLayer)
TopImageLayer = Dropout(0.20)(TopImageLayer)
#TopImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(TopImageLayer)
#TopImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(TopImageLayer)
#TopImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(TopImageLayer)
#TopImageLayer = Dropout(0.10)(TopImageLayer)

TopImageModel = Model(inputs = TopImageInputs, outputs = TopImageLayer)

#W Model
WImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )

WImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageInputs)
WImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = BatchNormalization(momentum = 0.6)(WImageLayer)
WImageLayer = MaxPool2D(pool_size=(2,2) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = BatchNormalization(momentum = 0.6)(WImageLayer)
WImageLayer = MaxPool2D(pool_size=(2,2) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
WImageLayer = BatchNormalization(momentum = 0.6)(WImageLayer)
WImageLayer = MaxPool2D(pool_size=(2,2) )(WImageLayer)
WImageLayer = Flatten()(WImageLayer)
WImageLayer = Dropout(0.20)(WImageLayer)
#WImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(WImageLayer)
#WImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(WImageLayer)
#WImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(WImageLayer)
#WImageLayer = Dropout(0.10)(WImageLayer)

WImageModel = Model(inputs = WImageInputs, outputs = WImageLayer)


#Z Model
ZImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )

ZImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageInputs)
ZImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = BatchNormalization(momentum = 0.6)(ZImageLayer)
ZImageLayer = MaxPool2D(pool_size=(2,2) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = BatchNormalization(momentum = 0.6)(ZImageLayer)
ZImageLayer = MaxPool2D(pool_size=(2,2) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
ZImageLayer = BatchNormalization(momentum = 0.6)(ZImageLayer)
ZImageLayer = MaxPool2D(pool_size=(2,2) )(ZImageLayer)
ZImageLayer = Flatten()(ZImageLayer)
ZImageLayer = Dropout(0.20)(ZImageLayer)
#ZImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
#ZImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
#ZImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
#ZImageLayer = Dropout(0.10)(ZImageLayer)

ZImageModel = Model(inputs = ZImageInputs, outputs = ZImageLayer)

# Create the BES variable version
besInputs = Input( shape=(BestShapeHolder, ) )
#besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besInputs)
#besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)

besModel = Model(inputs = besInputs, outputs = besInputs)
print (besModel.output)
# Add BES variables to the network
combined = concatenate([HiggsImageModel.output, TopImageModel.output, WImageModel.output, ZImageModel.output, besModel.output])
#Testing with just Higgs layer
#combined = concatenate([HiggsImageModel.output, TopImageModel.output, besModel.output])
#combined = concatenate([HiggsImageModel.output, TopImageModel.output, besInputs])

#combined = besModel.output
combLayer = Dense(512, kernel_initializer="glorot_normal", activation="relu" )(combined)
combLayer = Dropout(0.20)(combLayer)
combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dropout(0.10)(combLayer)
outputBEST = Dense(6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

# compile the model
model_BEST = Model(inputs = [HiggsImageModel.input, TopImageModel.input, WImageModel.input, ZImageModel.input, besModel.input], outputs = outputBEST)

model_BEST.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print the model summary
print(model_BEST.summary() )

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto', restore_best_weights=True)

# model checkpoint callback
# this saves the model architecture + parameters into dense_model.h5
model_checkpoint = ModelCheckpoint('BEST_model.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   period=1)

# train the neural network
history = model_BEST.fit([jetImagesTrain[:], jetBESvarsTrain[:] ], truthLabelsTrain[:], batch_size=1000, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_data = [[jetImagesValidation[:], jetBESvarsValidation[:]], truthLabelsValidation[:]])

print("Trained the neural network!")

# performance plots
loss = [history.history['loss'], history.history['val_loss'] ]
acc = [history.history['acc'], history.history['val_acc'] ]
tools.plotPerformance(loss, acc, "imageOnly")
print("plotted BEST training Performance")

print("Program was a great success!!!")
