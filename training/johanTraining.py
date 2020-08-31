#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# johanTraining.py ////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST with flattened inputs //////////////////////////////////
# One can train the netowork with only BESvars and images, or both ////////////////
# One can also ask to run the ensemble which takes/creates BES-only and //////////
# image-only networks, and feeds the output predictions into a separate network ///
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
import argparse, os
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import Sequential, Model
from keras.models import load_model
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
from plotConfusionMatrix import makeCM

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

# Do BES and/or images
doBES = False
doImages = False
doEnsemble = False
mySuffix = ""

sampleTypes = ["W","Z","Higgs","Top","b","QCD"]
frameTypes = ["W","Z","Higgs","Top"]

BatchSize = 1200

def loadData(setTypes):
   #==================================================================================
   # Initialize what will be np arrays ////////////////////////////////////////////////////////////
   #==================================================================================
   # This will create a series of global variables like jetTopFrameTrain and jetHiggsFrameValidation and jetBESvarsTrain, (4frames+1BesVars)*2sets=10globVars
   for mySet in setTypes:
      if doImages:
         for myFrame in frameTypes:
            globals()["jet"+myFrame+"Frame"+mySet] = []
      if doBES:
         globals()["jetBESvars"+mySet] = []

      globals()["truthLabels"+mySet] = []

   ## and this makes 12 global variables to store data

   print(globals().keys())

   #==================================================================================
   # Load Data from  h5 //////////////////////////////////////////////////////////////
   #==================================================================================

   # Loop over 2sets*6samples=12 files
   makeTruthLabelsOnce = True
   for mySet in setTypes:
      for index, mySample in enumerate(sampleTypes):
         print("Opening "+mySample+mySet+" file")
         myF = h5py.File("/uscms/home/bonillaj/nobackup/h5samples/"+mySample+"Sample_BESTinputs_"+mySet.lower()+"_flattened_standardized.h5","r")

         ## Make TruthLabels, only once (i.e. for key=BESvars)
         if globals()["truthLabels"+mySet] == []:
            print("Making new", "truthLabels"+mySet)
            globals()["truthLabels"+mySet] = numpy.full(len(myF['BES_vars'][()]), index)
         else:
            print("Concatenate", "truthLabels"+mySet)
            globals()["truthLabels"+mySet] = numpy.concatenate((globals()["truthLabels"+mySet], numpy.full(len(myF['BES_vars'][()]), index)))
            
         for myKey in myF.keys():
            varKey = "jet"
            if "image" in myKey.lower():
               if not doImages:
                  continue
               varKey = varKey+myKey.split("_")[0] # so HiggsFrame, TopFrame, etc
            else:
               if not doBES:
                  continue
               varKey = varKey+"BESvars"
               
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
   for mySet in setTypes:
      globals()["truthLabels"+mySet] = to_categorical(globals()["truthLabels"+mySet], num_classes = 6)
      print("Made Truth Labels "+mySet, globals()["truthLabels"+mySet].shape)


def train(doBES, doImages, outDir, suffix):
   
   if doBES:
      print("BESvars Train Shape", globals()["jetBESvarsTrain"].shape)
      print("BESvars Validation Shape", globals()["jetBESvarsValidation"].shape)
   for myFrame in frameTypes:
      if not doImages:
         continue
      print(myFrame+" Images Train Shape", globals()["jet"+myFrame+"FrameTrain"].shape)
      print(myFrame+" Images Validation Shape", globals()["jet"+myFrame+"FrameValidation"].shape)

   print("Shuffle Train")
   rng_state = numpy.random.get_state()
   numpy.random.set_state(rng_state)
   numpy.random.shuffle(globals()["truthLabelsTrain"])
   if doBES:
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBESvarsTrain"])
   if doImages:
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetWFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetZFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetHiggsFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetTopFrameTrain"])

   print("Shuffle Validation")
   numpy.random.set_state(rng_state)
   numpy.random.shuffle(globals()["truthLabelsValidation"])
   if doBES:
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBESvarsValidation"])
   if doImages:
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetWFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetZFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetHiggsFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetTopFrameValidation"])

   
   print("Stored data and truth information")

   #==================================================================================
   # Train the Neural Network ////////////////////////////////////////////////////////
   #==================================================================================
   # Shape parameters
   if doImages:
      arbitrary_length = 10 #Hopefully this number doesn't matter
      nx = 31
      ny = 31
      ImageShapeHolder = numpy.zeros((arbitrary_length, nx, ny, 1))

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
      ZImageLayer = Dropout(0.20)(ZImageLayer)#try 0.35 dropout here and in other images networks
      #ZImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
      #ZImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
      #ZImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
      #ZImageLayer = Dropout(0.10)(ZImageLayer)

      ZImageModel = Model(inputs = ZImageInputs, outputs = ZImageLayer)

   # Create the BES variable version
   if doBES:
      BestShapeHolder = 94
      besInputs = Input( shape=(BestShapeHolder, ) )
      #besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besInputs)
      #besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)
      
      besModel = Model(inputs = besInputs, outputs = besInputs)
      print (besModel.output)

   # Add BES variables to the network
   if doBES and not doImages:
      combined = besModel.output
   elif not doBES and doImages:
      combined = concatenate([WImageModel.output, ZImageModel.output, HiggsImageModel.output, TopImageModel.output])
   elif doBES and doImages:
      combined = concatenate([WImageModel.output, ZImageModel.output, HiggsImageModel.output, TopImageModel.output, besModel.output])

   combLayer = Dense(512, kernel_initializer="glorot_normal", activation="relu" )(combined)
   combLayer = Dropout(0.20)(combLayer)# try 0.35
   combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
   #Another dropout of 0.35
   combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
   combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
   combLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
   combLayer = Dropout(0.10)(combLayer)
   outputModel = Dense(6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

   # compile the model
   if doBES and not doImages:
      myModel = Model(inputs = [besModel.input], outputs = outputModel)
   elif not doBES and doImages:
      myModel = Model(inputs = [WImageModel.input, ZImageModel.input, HiggsImageModel.input, TopImageModel.input], outputs = outputModel)
   elif doBES and doImages:
      myModel = Model(inputs = [WImageModel.input, ZImageModel.input, HiggsImageModel.input, TopImageModel.input, besModel.input], outputs = outputModel)

   myModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   
   print(myModel.summary() )

   # early stopping
   early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto', restore_best_weights=True)


   # model checkpoint callback
   # this saves the model architecture + parameters into dense_model.h5
   
   model_checkpoint = ModelCheckpoint(outDir+'BEST_model'+suffix+'.h5', monitor='val_loss', 
                                      verbose=0, save_best_only=True, 
                                      save_weights_only=False, mode='auto', 
                                      period=1)

   # train the neural network
   if doBES and not doImages:
      history = myModel.fit([globals()["jetBESvarsTrain"][:] ], globals()["truthLabelsTrain"][:], batch_size=BatchSize, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_data = [[globals()["jetBESvarsValidation"][:]], globals()["truthLabelsValidation"][:]])
   elif not doBES and doImages:
      history = myModel.fit([globals()["jetWFrameTrain"][:], globals()["jetZFrameTrain"][:], globals()["jetHiggsFrameTrain"][:], globals()["jetTopFrameTrain"][:]], globals()["truthLabelsTrain"][:], batch_size=BatchSize, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_data = [[globals()["jetWFrameValidation"][:], globals()["jetZFrameValidation"][:], globals()["jetHiggsFrameValidation"][:], globals()["jetTopFrameValidation"][:]], globals()["truthLabelsValidation"][:]])
   elif doBES and doImages:
      history = myModel.fit([globals()["jetWFrameTrain"][:], globals()["jetZFrameTrain"][:], globals()["jetHiggsFrameTrain"][:], globals()["jetTopFrameTrain"][:], globals()["jetBESvarsTrain"][:] ], globals()["truthLabelsTrain"][:], batch_size=BatchSize, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_data = [[globals()["jetWFrameValidation"][:], globals()["jetZFrameValidation"][:], globals()["jetHiggsFrameValidation"][:], globals()["jetTopFrameValidation"][:], globals()["jetBESvarsValidation"][:]], globals()["truthLabelsValidation"][:]])

   print("Trained the neural network!")

   # performance plots
   loss = [history.history['loss'], history.history['val_loss'] ]
   acc = [history.history['acc'], history.history['val_acc'] ]
   tools.plotPerformance(loss, acc, suffix)
   print("plotted BEST training Performance")

   return myModel

def ensemble(model_BES, model_Images, suffix):

   ## PredictTrain1 should give an array of (NEvents, classification), BESvars
   ## PredictTrain2 should give an array of (NEvents, classification), Images
   ## Same for validation
   predictTrainBES = model_BES.predict([globals()["jetBESvarsTrain"][:]])
   predictTrainImages = model_Images.predict([globals()["jetWFrameTrain"][:], globals()["jetZFrameTrain"][:], globals()["jetHiggsFrameTrain"][:], globals()["jetTopFrameTrain"][:]])
   predictValidationBES = model_BES.predict([globals()["jetBESvarsValidation"][:]])
   predictValidationImages = model_Images.predict([globals()["jetWFrameValidation"][:], globals()["jetZFrameValidation"][:], globals()["jetHiggsFrameValidation"][:], globals()["jetTopFrameValidation"][:]])
   print("PredictTrainBES",predictTrainBES.shape, type(predictTrainBES))
   print("PredictTrainImages",predictTrainImages.shape, type(predictTrainImages))
   print("PredictValidationBES",predictValidationBES.shape, type(predictValidationBES))
   print("PredictTrainImages",predictValidationImages.shape, type(predictValidationImages))

   ## Need to make new network combining output of other networks here
   EnsembleShapeHolder = 12 #Six category weights for images and six for BES
   ensembleInputs = Input( shape=(EnsembleShapeHolder,) )
   ensembleModel = Model(inputs = ensembleInputs, outputs = ensembleInputs)
      
   ensemble = ensembleModel.output
   ensembleLayer = Dense(512, kernel_initializer="glorot_normal", activation="relu" )(ensemble)
   ensembleLayer = Dropout(0.20)(ensembleLayer)# try 0.35
   ensembleLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(ensembleLayer)
   #Another dropout of 0.35
   ensembleLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(ensembleLayer)
   ensembleLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(ensembleLayer)
   ensembleLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(ensembleLayer)
   ensembleLayer = Dropout(0.10)(ensembleLayer)
   outputEnsemble = Dense(6, kernel_initializer="glorot_normal", activation="softmax")(ensembleLayer)
      
   model_Ensemble = Model(inputs = [ensembleModel.input], outputs = outputEnsemble)
   model_Ensemble.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   print(model_Ensemble.summary() )
   # early stopping
   early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto', restore_best_weights=True)
   model_checkpointEnsemble = ModelCheckpoint('BEST_model'+suffix+'.h5', monitor='val_loss', 
                                              verbose=0, save_best_only=True, 
                                              save_weights_only=False, mode='auto', 
                                              period=1)
   
   concatTrain = numpy.concatenate((predictTrainBES[:], predictTrainImages[:]),axis=1)
   concatValidation = numpy.concatenate((predictValidationBES[:], predictValidationImages[:]),axis=1)
   print("concatTrain", concatTrain.shape, type(concatTrain))
   print("concatValidation", concatValidation.shape, type(concatValidation))
   print("truthLabelsTrain", globals()["truthLabelsTrain"].shape, type(globals()["truthLabelsTrain"]))
   print("truthLabelsValidation", globals()["truthLabelsValidation"].shape, type(globals()["truthLabelsValidation"]))
   historyEnsemble = model_Ensemble.fit([concatTrain], globals()["truthLabelsTrain"][:], batch_size=BatchSize, epochs=200, callbacks=[early_stopping, model_checkpointEnsemble], validation_data = [[concatValidation], globals()["truthLabelsValidation"][:]])
   lossEnsemble = [historyEnsemble.history['loss'], historyEnsemble.history['val_loss'] ]
   accEnsemble = [historyEnsemble.history['acc'], historyEnsemble.history['val_acc'] ]
   tools.plotPerformance(lossEnsemble, accEnsemble, suffix)

   print("Trained Ensembler")
   return model_Ensemble


# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
   # Take in arguments
   parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
   parser.add_argument('-hd','--h5Dir',
                       dest='h5Dir',
                       default="~/nobackup/h5samples/")
   parser.add_argument('-o','--outDir',
                       dest='outDir',
                       default="~/nobackup/models/")
   parser.add_argument('-sf','--suffix',
                       dest='suffix',
                       default="")
   parser.add_argument('-b','--doBES', dest='doBES', default=False, action='store_true')
   parser.add_argument('-i','--doImages', dest='doImages', default=False, action='store_true')
   parser.add_argument('-e','--doEnsemble', dest='doEnsemble', default=False, action='store_true')
   parser.add_argument('-r','--redoTraining', dest='redoTraining', default=False, action='store_true')
   args = parser.parse_args()
   
   if args.doBES: doBES = True
   if args.doImages: doImages = True
   if args.doEnsemble: doEnsemble = True
   if doEnsemble:
      doBES = True
      doImages = True
      mySuffix = mySuffix + "_Ensemble"
   else:
      if doBES and not doImages:
         mySuffix = mySuffix + "_BES"
      elif not doBES and doImages:
         mySuffix = mySuffix + "_Images"
      else:
         mySuffix = mySuffix + "_Both"
      
   # Make directories you need
   if not os.path.isdir(args.h5Dir):
      print(args.h5Dir, "does not exist")
      quit()
   if not os.path.isdir(args.outDir):
      print(args.outDir, "does not exist")
      quit()

   if args.redoTraining:
      print("Redo all training")
      loadData(["Train","Validation"])
      if doEnsemble:
         BES_model = train(doBES, False, args.outDir, "_BES")
         Images_model = train(False, doImages, args.outDir, "_Images")
         BEST_model = ensemble(BES_model, Images_model, mySuffix)
      else:
         BEST_model = train(doBES, doImages, args.outDir, mySuffix)
   else:
      print("Finding models available, training what is missing")
      if doEnsemble:
         if not os.path.isfile(args.outDir+"BEST_model_BES.h5"):
            print("Train BESvars model")
            if not "jetBESvarsTrain" in globals().keys():
               loadData(["Train","Validation"])
            BES_model = train(doBES, False, args.outDir, "_BES")
         else:
            print("Loading BES model")
            BES_model = load_model(args.outDir+"BEST_model_BES.h5")
         if not os.path.isfile(args.outDir+"BEST_model_Images.h5"):
            print("Train Image model")
            if not "jetWFrameTrain" in globals().keys():
               loadData(["Train","Validation"])
            Images_model = train(False, doImages, args.outDir, "_Images")
         else:
            print("Loading Images model")
            Images_model = load_model(args.outDir+"BEST_model_Images.h5")
         if not os.path.isfile(args.outDir+"BEST_model_Ensemble.h5"):
            print("Train Ensemble model")
            if not "jetBESvarsTrain" in globals().keys():
               loadData(["Train","Validation"])
            BEST_model = ensemble(BES_model, Images_model, mySuffix)
         else:
            print("Loading Ensemble model")
            BEST_model = load_model(args.outDir+"BEST_model_Ensemble.h5")
      else:
         BEST_model = None
         if doBES and not doImages:
            if not os.path.isfile(args.outDir+"BEST_model_BES.h5"):
               print("BESonly model not found, training a new one")
               if not "jetBESvarsTrain" in globals().keys():
                  loadData(["Train","Validation"])
            else:
               print("Loading BES-only model")
               BEST_model = load_model(args.outDir+"BEST_model_BES.h5")
         elif not doBES and doImages:
            if not os.path.isfile(args.outDir+"BEST_model_Images.h5"):
               print("Imagesonly model not found, training a new one")
               if not "jetWFrameTrain" in globals().keys():
                  loadData(["Train","Validation"])
            else:
               print("Loading Images-only model")
               BEST_model = load_model(args.outDir+"BEST_model_Images.h5")
         else:
            if not os.path.isfile(args.outDir+"BEST_model_Ensembler.h5"):
               print("Combined (both BESvars and Images, but not ensembled) model not found, training a new one")
               if not "jetBESvarsTrain" in globals().keys():
                  loadData(["Train","Validation"])
            else:
               print("Loading combined BES-images (no ensemble) model")
               BEST_model = load_model(args.outDir+"BEST_model_Combined.h5")
         if BEST_model == None:
            BEST_model = train(doBES, doImages, args.outDir, mySuffix)

   for mySet in ["Train","Validation"]:
      if doImages:
         for myFrame in frameTypes:
            if "jet"+myFrame+"Frame"+mySet in globals().keys():
               del globals()["jet"+myFrame+"Frame"+mySet]
      if doBES:
         if "jetBESvars"+mySet in globals().keys():
            del globals()["jetBESvars"+mySet]
         
   makeCM(BEST_model, doBES, doImages, doEnsemble, mySuffix)
