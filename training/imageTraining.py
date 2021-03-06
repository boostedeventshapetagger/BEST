#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# imageTraining.py ////////////////////////////////////////////////////////////////
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

#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
# put images in data frames
jetImagesDF = {}
QCD = h5py.File("../formatConverter/h5samples/QCDSample_BESTinputs.h5","r")
jetImagesDF['QCD'] = QCD['HiggsFrame_images'][()]
QCD.close()

HH = h5py.File("../formatConverter/h5samples/HiggsSample_BESTinputs.h5","r")
jetImagesDF['HH'] = HH['HiggsFrame_images'][()]
HH.close()

ZZ = h5py.File("../formatConverter/h5samples/ZSample_BESTinputs.h5","r")
jetImagesDF['ZZ'] = ZZ['HiggsFrame_images'][()]
ZZ.close()

WW = h5py.File("../formatConverter/h5samples/WSample_BESTinputs.h5","r")
jetImagesDF['WW'] = WW['HiggsFrame_images'][()]
WW.close()

tt = h5py.File("../formatConverter/h5samples/TopSample_BESTinputs.h5","r")
jetImagesDF['tt'] = tt['HiggsFrame_images'][()]
tt.close()

bb = h5py.File("../formatConverter/h5samples/bSample_BESTinputs.h5","r")
jetImagesDF['bb'] = bb['HiggsFrame_images'][()]
bb.close()

print("Accessed Jet Images")
print("Made image dataframes")

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Store data and truth
print("Number of QCD Jet Images: ", len(jetImagesDF['QCD']) )
print("Number of Higgs Jet Images: ", len(jetImagesDF['HH']) )
print("Number of W Jet Images: ", len(jetImagesDF['WW']) )
print("Number of Z Jet Images: ", len(jetImagesDF['ZZ']) )
print("Number of t Jet Images: ", len(jetImagesDF['tt']) )
print("Number of b Jet Images: ", len(jetImagesDF['bb']) )
jetImages = numpy.concatenate([jetImagesDF['WW'], jetImagesDF['ZZ'], jetImagesDF['HH'], jetImagesDF['tt'], jetImagesDF['bb'], jetImagesDF['QCD'] ])
jetLabels = numpy.concatenate([numpy.zeros(len(jetImagesDF['WW']) ), numpy.ones(len(jetImagesDF['ZZ']) ), numpy.full(len(jetImagesDF['HH']), 2),
                            numpy.full(len(jetImagesDF['tt']), 3), numpy.full(len(jetImagesDF['bb']), 4), numpy.full(len(jetImagesDF['QCD']), 5)] )

print("Stored data and truth information")

# split the training and testing data
trainData, testData, trainTruth, testTruth = train_test_split(jetImages, jetLabels, test_size=0.1)
#data_tuple = list(zip(trainData,trainTruth))
#random.shuffle(data_tuple)
#trainData, trainTruth = zip(*data_tuple)
#trainData=numpy.array(trainData)
#trainTruth=numpy.array(trainTruth)

print("Number of QCD jets in training: ", numpy.sum(trainTruth == 0) )
print("Number of H jets in training: ", numpy.sum(trainTruth == 1) )
print("Number of W jets in training: ", numpy.sum(trainTruth == 2) )

print("Number of QCD jets in testing: ", numpy.sum(testTruth == 0) )
print("Number of H jets in testing: ", numpy.sum(testTruth == 1) )
print("Number of W jets in testing: ", numpy.sum(testTruth == 2) )

# make it so keras results can go in a pkl file
#tools.make_keras_picklable()

# get the truth info in the correct form
trainTruth = to_categorical(trainTruth, num_classes=6)
testTruth = to_categorical(testTruth, num_classes=6)

# Define the Neural Network Structure
print("NN input shape: ", trainData.shape[1], trainData.shape[2], trainData.shape[3] )
#model_BESTNN = Sequential()
#model_BESTNN.add( Conv2D(12, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01), input_shape=(trainData.shape[1], trainData.shape[2], trainData.shape[3]) ))
#model_BESTNN.add( BatchNormalization(momentum = 0.6) )
#model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
#model_BESTNN.add( Conv2D(8, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
#model_BESTNN.add( BatchNormalization(momentum = 0.6) )
#model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
#model_BESTNN.add( Conv2D(8, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
#model_BESTNN.add( BatchNormalization(momentum = 0.6) )
#model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
#model_BESTNN.add( Flatten() )

#model_BESTNN.add( SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01), input_shape=(trainData.shape[1], trainData.shape[2], trainData.shape[3]) ))
#model_BESTNN.add( SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
#model_BESTNN.add( SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
#model_BESTNN.add( SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
#model_BESTNN.add( BatchNormalization(momentum = 0.6) )
#model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
#model_BESTNN.add( Dropout(0.10) )
#model_BESTNN.add( SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
#model_BESTNN.add( SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
#model_BESTNN.add( SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
#model_BESTNN.add( BatchNormalization(momentum = 0.6) )
#model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
#model_BESTNN.add( Dropout(0.10) )
#model_BESTNN.add( SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
#model_BESTNN.add( SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))

model_BESTNN = Sequential()
model_BESTNN.add( Conv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01), input_shape=(trainData.shape[1], trainData.shape[2], trainData.shape[3]) ))
model_BESTNN.add( Conv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( Conv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( Conv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( BatchNormalization(momentum = 0.6) )
model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
#model_BESTNN.add( Dropout(0.10) )
model_BESTNN.add( Conv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( Conv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( Conv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( BatchNormalization(momentum = 0.6) )
model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
#model_BESTNN.add( Dropout(0.10) )
model_BESTNN.add( Conv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( Conv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) ))
model_BESTNN.add( BatchNormalization(momentum = 0.6) )
#model_BESTNN.add( MaxPool2D(pool_size=(2,2) ) )
model_BESTNN.add( Flatten() )
model_BESTNN.add( Dropout(0.20) )
model_BESTNN.add( Dense(1000, kernel_initializer="glorot_normal", activation="relu" ))
model_BESTNN.add( Dense(1000, kernel_initializer="glorot_normal", activation="relu" ))
model_BESTNN.add( Dense(256, kernel_initializer="glorot_normal", activation="relu" ))
model_BESTNN.add( Dense(144, kernel_initializer="glorot_normal", activation="relu" ))
model_BESTNN.add( Dropout(0.10) )
model_BESTNN.add( Dense(6, kernel_initializer="glorot_normal", activation="softmax"))

# compile the model
model_BESTNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# print the model summary
print(model_BESTNN.summary() )

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')

# model checkpoint callback
# this saves the model architecture + parameters into dense_model.h5
model_checkpoint = ModelCheckpoint('BEST_imageOnly_model.h5', monitor='val_loss', 
                                   verbose=0, save_best_only=True, 
                                   save_weights_only=False, mode='auto', 
                                   period=1)

# train the neural network
history = model_BESTNN.fit(trainData[:], trainTruth[:], batch_size=1000, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_split = 0.15)

print("Trained the neural network!")

# save the test data
h5f = h5py.File("images/BESTimageOnlyTestData.h5","w")
h5f.create_dataset('test_images', data=testData, compression='lzf')
h5f.create_dataset('test_truth', data=testTruth, compression='lzf')

print("Saved the testing data!")

# print model visualization
#plot_model(model_BESTNN, to_file='plots/boost_CosTheta_NN_Vis.png')

#==================================================================================
# Plot Training Results ///////////////////////////////////////////////////////////
#==================================================================================

# Confusion Matrix
cm = metrics.confusion_matrix(numpy.argmax(model_BESTNN.predict(testData[:]), axis=1), numpy.argmax(testTruth[:], axis=1) )
plt.figure()
targetNames = ['W', 'Z', 'H', 't', 'b', 'QCD']
tools.plot_confusion_matrix(cm.T, targetNames, normalize=True)
if savePDF == True:
   plt.savefig('plots/imageOnly_confusion_matrix.pdf')
if savePNG == True:
   plt.savefig('plots/imageOnly_CosTheta_confusion_matrix.png')
plt.close()

# score
print("Training Score: ", model_BESTNN.evaluate(testData[:], testTruth[:], batch_size=100) )

# performance plots
loss = [history.history['loss'], history.history['val_loss'] ]
acc = [history.history['acc'], history.history['val_acc'] ]
tools.plotPerformance(loss, acc, "imageOnly")
print("plotted BEST training Performance")

# make file with probability results
#joblib.dump(model_BESTNN, "BEST_imageOnly_network.pkl")
#joblib.dump(scaler, "BEST_imageOnly_scaler.pkl")

print("Made weights based on probability results")
print("Program was a great success!!!")
