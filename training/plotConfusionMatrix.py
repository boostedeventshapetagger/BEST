#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plotConfusionMatrix.py ////////////////////////////////////////////////////////////////
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
import os
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
import tools.functions as functs

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

# set options 
savePDF = True
savePNG = True
doBES = True
doImages = True
doEnsembler = True
if doEnsembler:
   doBES = True
   doImages = True
suffix = '_Ensembler'

setTypes = ["Test"]
sampleTypes = ["W","Z","Higgs","Top","b","QCD"]
frameTypes = ["W","Z","Higgs","Top"]

BatchSize = 1200

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
      
#jetImagesTrain = [] #Should be a concatenation of XFrameImageTrain (ensure above sampleType order), each which appends {W,Z,H,Top,b,QCD}_XFrame_images_train
#jetImagesValidation = [] #Should be a concatenation of XFrameImageValidation (ensure above sampleType order), each which appends {W,Z,H,Top,b,QCD}_XFrame_images_train

truthLabelsTest = []

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

      ## Make TruthLabels, only once (i.e. for key=BESvars)
      if globals()["truthLabels"+mySet] == []:
         print("Making new", "truthLabels"+mySet)
         globals()["truthLabels"+mySet] = numpy.full(len(myF["BES_vars"][()]), index)
      else:
         print("Concatenate", "truthLabels"+mySet)
         globals()["truthLabels"+mySet] = numpy.concatenate((globals()["truthLabels"+mySet], numpy.full(len(myF["BES_vars"][()]), index)))
      
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
truthLabelsTest = to_categorical(truthLabelsTest, num_classes = 6)
print("Made Truth Labels Test", truthLabelsTest.shape)

## Concatenate Images: W, Z, Higgs, Top -> single
#print("Begin concatenating images")
#jetImagesTrain = numpy.concatenate([globals()["jet"+myFrame+"FrameTrain"] for myFrame in frameTypes])
#print("Concatenated training images", jetImagesTrain.shape)
#jetImagesValidation = numpy.concatenate([globals()["jet"+myFrame+"FrameValidation"] for myFrame in frameTypes])
#print("Concatenated validation images", jetImagesValidation.shape)
#print("Finished Image Concatenation")
if doBES:
   print("BESvars Test Shape", jetBESvarsTest.shape)
for myFrame in frameTypes:
   if not doImages:
      continue
   print(myFrame+" Images Train Shape", globals()["jet"+myFrame+"FrameTest"].shape)

print("Shuffle Test")
rng_state = numpy.random.get_state()
numpy.random.set_state(rng_state)
numpy.random.shuffle(truthLabelsTest)
if doBES:
   numpy.random.set_state(rng_state)
   numpy.random.shuffle(jetBESvarsTest)
if doImages:
   numpy.random.set_state(rng_state)
   #numpy.random.shuffle(jetImagesTrain)
   numpy.random.shuffle(jetWFrameTest)
   numpy.random.set_state(rng_state)
   numpy.random.shuffle(jetZFrameTest)
   numpy.random.set_state(rng_state)
   numpy.random.shuffle(jetHiggsFrameTest)
   numpy.random.set_state(rng_state)
   numpy.random.shuffle(jetTopFrameTest)

print("Load model")
cm = {}
if doEnsembler:
   model_BEST1 = load_model("BEST_model1"+suffix+".h5")
   model_BEST2 = load_model("BEST_model2"+suffix+".h5")
   predictTest1 = model_BEST1.predict([jetBESvarsTest[:]])
   predictTest2 = model_BEST2.predict([jetWFrameTest[:], jetZFrameTest[:], jetHiggsFrameTest[:], jetTopFrameTest[:]])
   model_BEST = load_model("BEST_modelEnsemble"+suffix+".h5")
   print("Make confusion matrix")
   cm["Ensemble1"] = metrics.confusion_matrix(numpy.argmax(model_BEST1.predict([jetBESvarsTest[:] ]), axis=1), numpy.argmax(truthLabelsTest[:], axis=1) )
   cm["Ensemble2"] = metrics.confusion_matrix(numpy.argmax(model_BEST2.predict([jetWFrameTest[:], jetZFrameTest[:], jetHiggsFrameTest[:], jetTopFrameTest[:]]), axis=1), numpy.argmax(truthLabelsTest[:], axis=1) )
   cm["EnsembleCombined"] = metrics.confusion_matrix(numpy.argmax(model_BEST.predict([numpy.concatenate((predictTest1[:], predictTest2[:]), axis=1)]), axis=1), numpy.argmax(truthLabelsTest[:], axis=1) )
else:
   model_BEST = load_model("BEST_model"+suffix+".h5")
   if doBES and not doImages:
      cm.append(metrics.confusion_matrix(numpy.argmax(model_BEST.predict([jetBESvarsTest[:] ]), axis=1), numpy.argmax(truthLabelsTest[:], axis=1) ))
   elif doBES and doImages:
      cm.append(metrics.confusion_matrix(numpy.argmax(model_BEST.predict([jetWFrameTest[:], jetZFrameTest[:], jetHiggsFrameTest[:], jetTopFrameTest[:], jetBESvarsTest[:] ]), axis=1), numpy.argmax(truthLabelsTest[:], axis=1) ))
   elif not doBES and doImages:
      cm.append(metrics.confusion_matrix(numpy.argmax(model_BEST.predict([jetWFrameTest[:], jetZFrameTest[:], jetHiggsFrameTest[:], jetTopFrameTest[:]]), axis=1), numpy.argmax(truthLabelsTest[:], axis=1) ))
print("Plot confusion matrix")
plt.figure(
)
targetNames = ['W', 'Z', 'Higgs', 'Top', 'b', 'QCD']
for myKey in cm.keys():
   functs.plot_confusion_matrix(cm[myKey].T, targetNames, normalize=True)
   if savePDF == True:
      if not os.path.isdir("plots"+suffix):
         os.mkdir("plots"+suffix)
      plt.savefig('plots'+suffix+'/ConfusionMatrix'+myKey+suffix+'.pdf')
   plt.close()


#loss = [history.history['loss'], history.history['val_loss'] ]
#acc = [history.history['acc'], history.history['val_acc'] ]
#functs.plotPerformance(loss, acc, "FlatPT")

print("Finished")
