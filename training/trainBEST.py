#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainBEST.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST: Boosted Event Shape Tagger ////////////////////////////
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

#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
# put images in data frames
jetImagesDF  = {}
jetBESvarsDF = {}

QCD = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/qcdBoostedJetImages.h5","r")
jetImagesDF['QCD']  = QCD['jet_images'][()]
jetBESvarsDF['QCD'] = QCD['BES_vars'][()]
QCD.close()

HH = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/HiggsBoostedJetImages.h5","r")
jetImagesDF['HH']  = HH['jet_images'][()]
jetBESvarsDF['HH'] = HH['BES_vars'][()]
HH.close()

ZZ = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/ZBoostedJetImages.h5","r")
jetImagesDF['ZZ']  = ZZ['jet_images'][()]
jetBESvarsDF['ZZ'] = ZZ['BES_vars'][()]
ZZ.close()

WW = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/WBoostedJetImages.h5","r")
jetImagesDF['WW']  = WW['jet_images'][()]
jetBESvarsDF['WW'] = WW['BES_vars'][()]
WW.close()

tt = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/topBoostedJetImages.h5","r")
jetImagesDF['tt']  = tt['jet_images'][()]
jetBESvarsDF['tt'] = tt['BES_vars'][()]
tt.close()

bb = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/bottomBoostedJetImages.h5","r")
jetImagesDF['bb']  = bb['jet_images'][()]
jetBESvarsDF['bb'] = bb['BES_vars'][()]
bb.close()

print "Accessed Jet Images and BES variables"

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Store data and truth
print "Number of QCD Jet Images: ", len(jetImagesDF['QCD'])
print "Number of Higgs Jet Images: ", len(jetImagesDF['HH'])
print "Number of W Jet Images: ", len(jetImagesDF['WW'])
print "Number of Z Jet Images: ", len(jetImagesDF['ZZ'])
print "Number of t Jet Images: ", len(jetImagesDF['tt'])
print "Number of b Jet Images: ", len(jetImagesDF['bb'])
jetBESvars = numpy.concatenate([jetBESvarsDF['WW'], jetBESvarsDF['ZZ'], jetBESvarsDF['HH'], jetBESvarsDF['tt'], jetBESvarsDF['bb'], jetBESvarsDF['QCD'] ])
jetImages  = numpy.concatenate([jetImagesDF['WW'], jetImagesDF['ZZ'], jetImagesDF['HH'], jetImagesDF['tt'], jetImagesDF['bb'], jetImagesDF['QCD'] ])
jetLabels  = numpy.concatenate([numpy.zeros(len(jetImagesDF['WW']) ), numpy.ones(len(jetImagesDF['ZZ']) ), numpy.full(len(jetImagesDF['HH']), 2),
                            numpy.full(len(jetImagesDF['tt']), 3), numpy.full(len(jetImagesDF['bb']), 4), numpy.full(len(jetImagesDF['QCD']), 5)] )

print "Stored data and truth information"

# Normalize the BES variables
scaler = preprocessing.StandardScaler().fit(jetBESvars)
jetBESvars = scaler.transform(jetBESvars)

# split the training and testing data
trainImages, testImages, trainBESvars, testBESvars, trainTruth, testTruth = train_test_split(jetImages, jetBESvars, jetLabels, test_size=0.1)
#data_tuple = list(zip(trainImages,trainTruth))
#random.shuffle(data_tuple)
#trainImages, trainTruth = zip(*data_tuple)
#trainImages=numpy.array(trainImages)
#trainTruth=numpy.array(trainTruth)

print "Number of W jets in training: ", numpy.sum(trainTruth == 0)

print "Number of W jets in testing: ", numpy.sum(testTruth == 0)

# make it so keras results can go in a pkl file
#tools.make_keras_picklable()

# get the truth info in the correct form
trainTruth = to_categorical(trainTruth, num_classes=6)
testTruth = to_categorical(testTruth, num_classes=6)

print "NN image input shape: ", trainImages.shape[1], trainImages.shape[2], trainImages.shape[3]

# Define the Neural Network Structure using functional API
# Create the image portion
imageInputs = Input( shape=(trainImages.shape[1], trainImages.shape[2], trainImages.shape[3]) )

imageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageInputs)
imageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = BatchNormalization(momentum = 0.6)(imageLayer)
imageLayer = MaxPool2D(pool_size=(2,2) )(imageLayer)
imageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = BatchNormalization(momentum = 0.6)(imageLayer)
imageLayer = MaxPool2D(pool_size=(2,2) )(imageLayer)
imageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(imageLayer)
imageLayer = BatchNormalization(momentum = 0.6)(imageLayer)
imageLayer = MaxPool2D(pool_size=(2,2) )(imageLayer)
imageLayer = Flatten()(imageLayer)
imageLayer = Dropout(0.20)(imageLayer)
#imageLayer = Dense(1000, kernel_initializer="glorot_normal", activation="relu" )(imageLayer)
#imageLayer = Dense(1000, kernel_initializer="glorot_normal", activation="relu" )(imageLayer)
#imageLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(imageLayer)
#imageLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(imageLayer)
#imageLayer = Dropout(0.10)(imageLayer)

imageModel = Model(inputs = imageInputs, outputs = imageLayer)

# Create the BES variable version
besInputs = Input( shape=(trainBESvars.shape[1], ) )
#besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besInputs)
#besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)
#besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)

besModel = Model(inputs = besInputs, outputs = besInputs)

# Add BES variables to the network
combined = concatenate([imageModel.output, besModel.output])

combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combined)
#combLayer = Dense(500, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
combLayer = Dropout(0.10)(combLayer)
outputBEST = Dense(6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

# compile the model
model_BEST = Model(inputs = [imageModel.input, besModel.input], outputs = outputBEST)
model_BEST.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print the model summary
print(model_BEST.summary() )

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')

# model checkpoint callback
# this saves the model architecture + parameters into dense_model.h5
model_checkpoint = ModelCheckpoint('BEST_model.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   period=1)

# train the neural network
history = model_BEST.fit([trainImages[:], trainBESvars[:] ], trainTruth[:], batch_size=1000, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_split = 0.15)

print "Trained the neural network!"

# print model visualization
#plot_model(model_BEST, to_file='plots/boost_CosTheta_NN_Vis.png')

# save the test data
h5f = h5py.File("images/BESTtestData.h5","w")
h5f.create_dataset('test_images', data=testImages, compression='lzf')
h5f.create_dataset('test_BES_vars', data=testBESvars, compression='lzf')
h5f.create_dataset('test_truth', data=testTruth, compression='lzf')

print "Saved the testing data!"


#==================================================================================
# Plot Training Results ///////////////////////////////////////////////////////////
#==================================================================================

# Confusion Matrix
cm = metrics.confusion_matrix(numpy.argmax(model_BEST.predict([testImages[:], testBESvars[:] ]), axis=1), numpy.argmax(testTruth[:], axis=1) )
plt.figure()
targetNames = ['W', 'Z', 'H', 't', 'b', 'QCD']
tools.plot_confusion_matrix(cm.T, targetNames, normalize=True)
if savePDF == True:
   plt.savefig('plots/BEST_confusion_matrix.pdf')
if savePNG == True:
   plt.savefig('plots/BEST_confusion_matrix.png')
plt.close()

# score
print "Training Score: ", model_BEST.evaluate([testImages[:], testBESvars[:]], testTruth[:], batch_size=100)

# performance plots
loss = [history.history['loss'], history.history['val_loss'] ]
acc = [history.history['acc'], history.history['val_acc'] ]
tools.plotPerformance(loss, acc, "BEST")
print "plotted BEST training Performance"

# make file with probability results
#joblib.dump(model_BEST, "BEST_keras_CosTheta.pkl")
#joblib.dump(scaler, "BEST_scaler.pkl")

print "Made weights based on probability results"
print "Program was a great success!!!"
