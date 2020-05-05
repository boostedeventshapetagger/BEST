#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainBEST.py /////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST: HH Event Shape Topology Indentification Algorithm //
#==================================================================================

# modules
import numpy as np
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
jetBESvarsDF = {}

QCD = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/qcdBoostedJetImages.h5","r")
jetBESvarsDF['QCD'] = QCD['BES_vars'][()]
QCD.close()

HH = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/HiggsBoostedJetImages.h5","r")
jetBESvarsDF['HH'] = HH['BES_vars'][()]
HH.close()

ZZ = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/ZBoostedJetImages.h5","r")
jetBESvarsDF['ZZ'] = ZZ['BES_vars'][()]
ZZ.close()

WW = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/WBoostedJetImages.h5","r")
jetBESvarsDF['WW'] = WW['BES_vars'][()]
WW.close()

tt = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/topBoostedJetImages.h5","r")
jetBESvarsDF['tt'] = tt['BES_vars'][()]
tt.close()

bb = h5py.File("/uscms_data/d3/bregnery/BEST/CMSSW_10_1_7/src/BEST/jetCamera/images/bottomBoostedJetImages.h5","r")
jetBESvarsDF['bb'] = bb['BES_vars'][()]
bb.close()

print "Accessed Jet Images and BES variables"


#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Store data and truth
print "Number of QCD Jet BESvars: ", len(jetBESvarsDF['QCD'])
print "QCD NaNs: ", jetBESvarsDF['QCD'][np.argwhere(np.isnan(jetBESvarsDF['QCD']))] 
print "Number of Higgs Jet BESvars: ", len(jetBESvarsDF['HH'])
print "H NaNs: ", len(np.argwhere(np.isnan(jetBESvarsDF['HH'])) )
print "Number of W Jet BESvars: ", len(jetBESvarsDF['WW'])
print "W NaNs: ", len(np.argwhere(np.isnan(jetBESvarsDF['WW'])) )
print "Number of Z Jet BESvars: ", len(jetBESvarsDF['ZZ'])
print "Z NaNs: ", len(np.argwhere(np.isnan(jetBESvarsDF['ZZ'])) )
print "Number of t Jet BESvars: ", len(jetBESvarsDF['tt'])
print "tt NaNs: ", len(np.argwhere(np.isnan(jetBESvarsDF['tt'])) )
print "Number of b Jet BESvars: ", len(jetBESvarsDF['bb'])
print "bb NaNs: ", len(np.argwhere(np.isnan(jetBESvarsDF['bb'])) )
jetBESvars = np.concatenate([jetBESvarsDF['WW'], jetBESvarsDF['ZZ'], jetBESvarsDF['HH'], jetBESvarsDF['tt'], jetBESvarsDF['bb'], jetBESvarsDF['QCD'] ])
jetLabels  = np.concatenate([np.zeros(len(jetBESvarsDF['WW']) ), np.ones(len(jetBESvarsDF['ZZ']) ), np.full(len(jetBESvarsDF['HH']), 2),
                            np.full(len(jetBESvarsDF['tt']), 3), np.full(len(jetBESvarsDF['bb']), 4), np.full(len(jetBESvarsDF['QCD']), 5)] )

print "Stored data and truth information"

# Normalize the BES variables
scaler = preprocessing.StandardScaler().fit(jetBESvars)
jetBESvars = scaler.transform(jetBESvars)

# split the training and testing data
trainBESvars, testBESvars, trainTruth, testTruth = train_test_split(jetBESvars, jetLabels, test_size=0.1)

# make it so keras results can go in a pkl file
#tools.make_keras_picklable()

# get the truth info in the correct form
trainTruth = to_categorical(trainTruth, num_classes=6)
testTruth = to_categorical(testTruth, num_classes=6)

# Define the Neural Network Structure using functional API
# Create the BES variable version
besInputs = Input( shape=(trainBESvars.shape[1], ) )
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besInputs)
#besLayer = Dropout(0.20)(besLayer)
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)
#besLayer = Dropout(0.20)(besLayer)
besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)
#besLayer = Dropout(0.20)(besLayer)
outputBEST = Dense(6, kernel_initializer="glorot_normal", activation="softmax")(besLayer)

# compile the model
model_BEST = Model(inputs = besInputs, outputs = outputBEST)
model_BEST.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print the model summary
print(model_BEST.summary() )

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')

# model checkpoint callback
# this saves the model architecture + parameters into dense_model.h5
model_checkpoint = ModelCheckpoint('BEST_BESonly_model.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   period=1)

# train the neural network
history = model_BEST.fit(trainBESvars[:], trainTruth[:], batch_size=1000, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_split = 0.15)

print "Trained the neural network!"

# save the test data
h5f = h5py.File("images/BESonly_BESTtestData.h5","w")
h5f.create_dataset('test_BES_vars', data=testBESvars, compression='lzf')
h5f.create_dataset('test_truth', data=testTruth, compression='lzf')

print "Saved the testing data!"


# print model visualization
#plot_model(model_BEST, to_file='plots/boost_CosTheta_NN_Vis.png')

#==================================================================================
# Plot Training Results ///////////////////////////////////////////////////////////
#==================================================================================

# Confusion Matrix
cm = metrics.confusion_matrix(np.argmax(model_BEST.predict(testBESvars[:] ), axis=1), np.argmax(testTruth[:], axis=1) )
plt.figure()
targetNames = ['W', 'Z', 'H', 't', 'b', 'QCD']
tools.plot_confusion_matrix(cm.T, targetNames, normalize=True)
if savePDF == True:
   plt.savefig('plots/BEST_BESonly_confustion_matrix.pdf')
if savePNG == True:
   plt.savefig('plots/BEST_BESonly_confusion_matrix.png')
plt.close()

# score
print "Training Score: ", model_BEST.evaluate(testBESvars[:], testTruth[:], batch_size=100)

# performance plots
loss = [history.history['loss'], history.history['val_loss'] ]
acc = [history.history['acc'], history.history['val_acc'] ]
tools.plotPerformance(loss, acc, "BEST_BESonly")
print "plotted HESTIA training Performance"

# make file with probability results
#joblib.dump(model_BEST, "BEST_keras_BESonly.pkl")
#joblib.dump(scaler, "BEST_scaler.pkl")

print "Made weights based on probability results"
print "Program was a great success!!!"
