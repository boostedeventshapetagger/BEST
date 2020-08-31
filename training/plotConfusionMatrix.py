#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plotConfusionMatrix.py //////////////////////////////////////////////////////////
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

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

# set options 
savePDF = True
savePNG = True

setTypes = ["Test"]
sampleTypes = ["W","Z","Higgs","Top","b","QCD"]
frameTypes = ["W","Z","Higgs","Top"]

BatchSize = 1200

models = ["BES","Images","Combined","Both"]

def loadData(setTypes, doBES, doImages):
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



'''
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
'''

def makeCM(model_BEST, doBES, doImages, doEnsemble, suffix):
   import tools.functions as functs
   #from johanTraining import loadData
   print("Before load")
   print(globals().keys())
   loadData(["Test"], doBES, doImages)
   print("After load")
   print(globals().keys())
   if doBES:
      print("BESvars Test Shape", globals()["jetBESvarsTest"].shape)
   for myFrame in frameTypes:
      if not doImages:
         continue
      print(myFrame+" Images Train Shape", globals()["jet"+myFrame+"FrameTest"].shape)

   print("Shuffle Test")
   rng_state = numpy.random.get_state()
   numpy.random.set_state(rng_state)
   numpy.random.shuffle(globals()["truthLabelsTest"])
   if doBES:
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBESvarsTest"])
   if doImages:
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetWFrameTest"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetZFrameTest"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetHiggsFrameTest"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetTopFrameTest"])

   print("Load model")
   cm = {}
   if doEnsemble:
      model_BES = load_model("/uscms/home/bonillaj/nobackup/models/BEST_model_BES.h5")
      model_Images = load_model("/uscms/home/bonillaj/nobackup/models/BEST_model_Images.h5")
      predictTestBES = model_BES.predict([globals()["jetBESvarsTest"][:]])
      predictTestImages = model_Images.predict([globals()["jetWFrameTest"][:], globals()["jetZFrameTest"][:], globals()["jetHiggsFrameTest"][:], globals()["jetTopFrameTest"][:]])
      #model_BEST = load_model(modelBEST)
      print("Make confusion matrix")
      cm["BES"] = metrics.confusion_matrix(numpy.argmax(model_BES.predict([globals()["jetBESvarsTest"][:] ]), axis=1), numpy.argmax(globals()["truthLabelsTest"][:], axis=1) )
      cm["Images"] = metrics.confusion_matrix(numpy.argmax(model_Images.predict([globals()["jetWFrameTest"][:], globals()["jetZFrameTest"][:], globals()["jetHiggsFrameTest"][:], globals()["jetTopFrameTest"][:]]), axis=1), numpy.argmax(globals()["truthLabelsTest"][:], axis=1) )
      cm["Ensemble"] = metrics.confusion_matrix(numpy.argmax(model_BEST.predict([numpy.concatenate((predictTestBES[:], predictTestImages[:]), axis=1)]), axis=1), numpy.argmax(globals()["truthLabelsTest"][:], axis=1) )
   else:
      #model_BEST = load_model(modelBEST)
      if doBES and not doImages:
         cm["BES"] = metrics.confusion_matrix(numpy.argmax(model_BEST.predict([globals()["jetBESvarsTest"][:] ]), axis=1), numpy.argmax(globals()["truthLabelsTest"][:], axis=1) )
      elif not doBES and doImages:
         cm["Images"] = metrics.confusion_matrix(numpy.argmax(model_BEST.predict([globals()["jetWFrameTest"][:], globals()["jetZFrameTest"][:], globals()["jetHiggsFrameTest"][:], globals()["jetTopFrameTest"][:]]), axis=1), numpy.argmax(globals()["truthLabelsTest"][:], axis=1) )
      elif doBES and doImages:
         cm["Combined"] = metrics.confusion_matrix(numpy.argmax(model_BEST.predict([globals()["jetWFrameTest"][:], globals()["jetZFrameTest"][:], globals()["jetHiggsFrameTest"][:], globals()["jetTopFrameTest"][:], globals()["jetBESvarsTest"][:] ]), axis=1), numpy.argmax(globals()["truthLabelsTest"][:], axis=1) )
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
      plt.clf()
   plt.close()

   print("Finished")


if __name__ == "__main__":
   from johanTraining import loadData
   loadData(["Test"])
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
   parser.add_argument('-m', '--models',
                        dest='models',
                        help='<Required> Which (comma separated) models to process. Examples: 1) all, 2) BES,Images,Combined,Ensemble',
                        required=True)
   if not args.samples == "all": listOfSamples = args.samples.split(',')
   parser.add_argument('-b','--doBES', dest='doBES', default=False, action='store_true')
   parser.add_argument('-i','--doImages', dest='doImages', default=False, action='store_true')
   parser.add_argument('-e','--doEnsemble', dest='doEnsemble', default=False, action='store_true')
   args = parser.parse_args()

   doBES = args.doBES
   doImages = args.doImages
   doEnsemble = args.doEnsemble
   if doEnsemble:
      doBES = True
      doImages = True
   for myModel in models:
      makeCM(load_model("BEST_model_"+myModel+".h5"), doBES, doImages, doEnsemble, args.suffix)
