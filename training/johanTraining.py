#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# johanTraining.py ////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST with flattened inputs //////////////////////////////////
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
import numpy.random

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
from keras.layers import concatenate
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

setTypes = ["Train","Validation"]
sampleTypes = ["W","Z","Higgs","Top","b","QCD"]
frameTypes = ["W","Z","Higgs","Top"]

BatchSize = 1200

#==================================================================================
# Initialize what will be np arrays ////////////////////////////////////////////////////////////
#==================================================================================
# This will create a series of global variables like jetTopFrameTrain and jetHiggsFrameValidation and jetBESvarsTrain, (4frames+1BesVars)*2sets=10globVars
for mySet in setTypes:
   for myFrame in frameTypes:
      globals()["jet"+myFrame+"Frame"+mySet] = []
   globals()["jetBESvars"+mySet] = []
      
#jetImagesTrain = [] #Should be a concatenation of XFrameImageTrain (ensure above sampleType order), each which appends {W,Z,H,Top,b,QCD}_XFrame_images_train
#jetImagesValidation = [] #Should be a concatenation of XFrameImageValidation (ensure above sampleType order), each which appends {W,Z,H,Top,b,QCD}_XFrame_images_train

truthLabelsTrain = []
truthLabelsValidation = []

## and this makes 12 global variables to store data

print(globals())


#==================================================================================
# Load Data from  h5 //////////////////////////////////////////////////////////////
#==================================================================================

# Loop over 2sets*6samples=12 files
for mySet in setTypes:
   for index, mySample in enumerate(sampleTypes):
      print("Opening "+mySample+mySet+" file")
      myF = h5py.File("/uscms/home/bonillaj/nobackup/h5samples/"+mySample+"Sample_BESTinputs_"+mySet.lower()+"_flattened_standardized.h5","r")
      for myKey in myF.keys():
         varKey = "jet"
         if "image" in myKey.lower():
            varKey = varKey+myKey.split("_")[0] # so HiggsFrame, TopFrame, etc
         else:
            varKey = varKey+"BESvars"
         
            ## Make TruthLabels, only once (i.e. for key=BESvars)
            if globals()["truthLabels"+mySet] == []:
               print("Making new", "truthLabels"+mySet)
               globals()["truthLabels"+mySet] = numpy.full(len(myF[myKey][()]), index)
            else:
               print("Concatenate", "truthLabels"+mySet)
               globals()["truthLabels"+mySet] = numpy.concatenate((globals()["truthLabels"+mySet], numpy.full(len(myF[myKey][()]), index)))
               
         varKey = varKey+mySet
         
         ## Append data
         if globals()[varKey] == []:
            print("Making new", varKey)
            globals()[varKey] = myF[myKey][()]
         else:
            print("Concatenate", varKey)
            globals()[varKey] = numpy.concatenate((globals()[varKey], myF[myKey][()]))
            
      myF.close()
      
print("Finished Accessing H5 data")

## Order of categories: 0-W, 1-Z, 2-H, 3-t, 4-b, 5-QCD (order of sampleTypes). Format properly.
print("To_Categorical")
truthLabelsTrain = to_categorical(truthLabelsTrain, num_classes = 6)
print("Made Truth Labels Train", truthLabelsTrain.shape)
truthLabelsValidation = to_categorical(truthLabelsValidation, num_classes = 6)
print("Made Truth Labels Validation", truthLabelsValidation.shape)

## Concatenate Images: W, Z, Higgs, Top -> single
#print("Begin concatenating images")
#jetImagesTrain = numpy.concatenate([globals()["jet"+myFrame+"FrameTrain"] for myFrame in frameTypes])
#print("Concatenated training images", jetImagesTrain.shape)
#jetImagesValidation = numpy.concatenate([globals()["jet"+myFrame+"FrameValidation"] for myFrame in frameTypes])
#print("Concatenated validation images", jetImagesValidation.shape)
#print("Finished Image Concatenation")
print("BESvars Train Shape", jetBESvarsTrain.shape)
print("BESvars Validation Shape", jetBESvarsValidation.shape)
for myFrame in frameTypes:
   print(myFrame+" Images Train Shape", globals()["jet"+myFrame+"FrameTrain"].shape)
   print(myFrame+" Images Validation Shape", globals()["jet"+myFrame+"FrameValidation"].shape)

print("Shuffle Train")
rng_state = numpy.random.get_state()
numpy.random.set_state(rng_state)
numpy.random.shuffle(truthLabelsTrain)
numpy.random.set_state(rng_state)
#numpy.random.shuffle(jetImagesTrain)
numpy.random.shuffle(jetWFrameTrain)
numpy.random.set_state(rng_state)
numpy.random.shuffle(jetZFrameTrain)
numpy.random.set_state(rng_state)
numpy.random.shuffle(jetHiggsFrameTrain)
numpy.random.set_state(rng_state)
numpy.random.shuffle(jetTopFrameTrain)
numpy.random.set_state(rng_state)
numpy.random.shuffle(jetBESvarsTrain)

print("Shuffle Validation")
numpy.random.set_state(rng_state)
numpy.random.shuffle(truthLabelsValidation)
numpy.random.set_state(rng_state)
#numpy.random.shuffle(jetImagesValidation)
numpy.random.shuffle(jetWFrameValidation)
numpy.random.set_state(rng_state)
numpy.random.shuffle(jetZFrameValidation)
numpy.random.set_state(rng_state)
numpy.random.shuffle(jetHiggsFrameValidation)
numpy.random.set_state(rng_state)
numpy.random.shuffle(jetTopFrameValidation)
numpy.random.set_state(rng_state)
numpy.random.shuffle(jetBESvarsValidation)


print("Stored data and truth information")

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================
# Shape parameters
arbitrary_length = 10 #Hopefully this number doesn't matter
nx = 31
ny = 31
ImageShapeHolder = numpy.zeros((arbitrary_length, nx, ny, 1))
BestShapeHolder = 94

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
combined = concatenate([WImageModel.output, ZImageModel.output, HiggsImageModel.output, TopImageModel.output, besModel.output])
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
model_BEST = Model(inputs = [WImageModel.input, ZImageModel.input, HiggsImageModel.input, TopImageModel.input, besModel.input], outputs = outputBEST)

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
history = model_BEST.fit([jetWFrameTrain[:], jetZFrameTrain[:], jetHiggsFrameTrain[:], jetTopFrameTrain[:], jetBESvarsTrain[:] ], truthLabelsTrain[:], batch_size=BatchSize, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_data = [[jetWFrameValidation[:], jetZFrameValidation[:], jetHiggsFrameValidation[:], jetTopFrameValidation[:], jetBESvarsValidation[:]], truthLabelsValidation[:]])

print("Trained the neural network!")

# performance plots
loss = [history.history['loss'], history.history['val_loss'] ]
acc = [history.history['acc'], history.history['val_acc'] ]
tools.plotPerformance(loss, acc, "newAxes")
print("plotted BEST training Performance")

print("Program was a great success!!!")
