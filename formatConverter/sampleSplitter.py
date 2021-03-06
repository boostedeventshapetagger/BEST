#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# sampleSplitter.py /////////////////////////////////////////////////////
#==================================================================================
# Author(s): Johan S Bonilla, Brendan Regnery -------------------------------------
# This program splits h5f files into 3 smaller but equal orthogonal files       ///
# The point is to create separate train/validation/test samples of equal size   ///
# Inputs should be h5f files (can be flatted, or staright from formatConverter) ///
# Output should be three sets of hd5f files: training, validation, testing      ///
#----------------------------------------------------------------------------------

# modules
import ROOT as root
import uproot
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import argparse
import os
import random
import time
from sklearn.model_selection import train_test_split

# User modules
import tools.imageOperations as img

# Enter batch mode in root, so python can access displays
root.gROOT.SetBatch(True)

# Global variables
listOfSamples = ["b","Higgs","QCD","Top","W","Z"]

# Helper functions
def splitFileSKL(inputPath, outDir, debug, userBatchSize):
    print("Starting clock")
    startTime = time.time()
    
    listOfFrameTypes = ["Higgs","Top","W","Z"]
    setTypes = ["train", "validation", "test"]

    # Open file, grab keys, and NEvents
    inputFile = h5py.File(inputPath,"r")
    dataKeys = inputFile.keys()
    totalEvents = inputFile[dataKeys[0]].shape[0]

    # Create data frame and output files to handle copied information
    # Make h5f file to store the images and BES variables
    outName = outDir+inputPath.split('.')[-2].split('/')[-1]
    h5fTrain = h5py.File(outName+"_train.h5","w")
    h5fValidation = h5py.File(outName+"_validation.h5","w")
    h5fTest = h5py.File(outName+"_test.h5","w")
        
    besData = {}
    outputFiles = {}
    for setType in setTypes:
        besData[setType] = {}

    counter = 0
    while (counter < totalEvents):
        batchTime = time.time()
        batchSize = userBatchSize if (totalEvents > counter+userBatchSize) else (totalEvents-counter)
        print("Batch size of ",batchSize,", at counter ",counter)
        print("Starting key loop")
        for myKey in dataKeys:
            keyTime = time.time()
            #dsetH5 = inputFile[myKey]
            dsetNP = np.array(inputFile[myKey][counter:counter+batchSize])
            print("NPdset creation time:", time.time()-keyTime)
            ## Shuffle=True shuffles the incoming data set.
            ## Random state sets the seed. The values are meaningless, but the same value leads to same results
            ## The most important thing here is that each key is passed the same random state so the same events are split and kept
            output1 = train_test_split(dsetNP, train_size=0.34, shuffle=True, random_state=42)
            output2 = train_test_split(output1[1], train_size=0.5, shuffle=True, random_state=24)
            print("Outdset creation time:", time.time()-keyTime)
            if counter == 0:
                if "frame" in myKey.lower():
                    besData["train"][myKey] = h5fTrain.create_dataset(myKey, data=output1[0], maxshape=(None, inputFile[myKey].shape[1], inputFile[myKey].shape[2], inputFile[myKey].shape[3]), compression='lzf')
                    besData["validation"][myKey] = h5fValidation.create_dataset(myKey, data=output2[0], maxshape=(None, inputFile[myKey].shape[1], inputFile[myKey].shape[2], inputFile[myKey].shape[3]), compression='lzf')
                    besData["test"][myKey] = h5fTest.create_dataset(myKey, data=output2[1], maxshape=(None, inputFile[myKey].shape[1], inputFile[myKey].shape[2], inputFile[myKey].shape[3]), compression='lzf')
                    print("DS store time:", time.time()-keyTime)
                else:
                    besData["train"][myKey] = h5fTrain.create_dataset(myKey, data=output1[0], maxshape=(None, inputFile[myKey].shape[1]), compression='lzf')
                    besData["validation"][myKey] = h5fValidation.create_dataset(myKey, data=output2[0], maxshape=(None, inputFile[myKey].shape[1]), compression='lzf')
                    besData["test"][myKey] = h5fTest.create_dataset(myKey, data=output2[1], maxshape=(None, inputFile[myKey].shape[1]), compression='lzf')
                    print("DS store time:", time.time()-keyTime)
            else:
                # append the dataset
                besData["train"][myKey].resize(besData["train"][myKey].shape[0] + len(output1[0]), axis=0)
                besData["train"][myKey][-len(output1[0]) :] = output1[0]
                besData["validation"][myKey].resize(besData["validation"][myKey].shape[0] + len(output2[0]), axis=0)
                besData["validation"][myKey][-len(output2[0]) :] = output2[0]
                besData["test"][myKey].resize(besData["test"][myKey].shape[0] + len(output2[1]), axis=0)
                besData["test"][myKey][-len(output2[1]) :] = output2[1] 
                print("DS store time:", time.time()-keyTime)
            print("Key iteration time:", time.time()-keyTime)
            keyTime = time.time()
        print("Batch time:", time.time()-batchTime)
        counter += batchSize
    print("Splitting time:", time.time()-startTime)

def splitFileSlow(inputPath, outDir, debug):
    listOfFrameTypes = ["Higgs","Top","W","Z"]
    setTypes = ["train", "validation", "test"]

    # Open file, grab keys and NEvents
    inputFile = h5py.File(inputPath,"r")
    dataKeys = inputFile.keys()
    totalEvents = inputFile[dataKeys[0]].shape[0]

    # Create data frame and output files to handle copied information
    besData = {}
    outputFiles = {}
    for setType in setTypes:
        besData[setType] = {}
        outputFiles[setType] = h5py.File(outDir+inputPath.split('.')[-2].split('/')[-1]+"_"+setType+".h5","w")

    startTime = time.time()

    # Create an array of size NEvents with randomized int elements
    # Entries 0,1,2 corresponds to train, validation, test (index in setTypes)
    split = np.random.randint(3, size=totalEvents)

    # Loop over keys in data: besVars, 4xImageFrames
    for thisKey in dataKeys:
        keyTime = time.time()
        print("Spltting Key: "+thisKey)

        # Create var pointing to the input data, by key.
        # Shape is (NEvents, NBESvars) or (NEvents, 31,3 1, 1) 
        keyData = inputFile[thisKey]
        if debug: print("Key Data", keyData.shape)

        # Loop over number of sets: train, valid, test
        for i in range(0,3):
            setTime = time.time()
            print("Creating "+setTypes[i]+" set")

            # Create random mask for set. Split is int array of size NEvents
            # Operating with bool on entire array
            # Output is bool array of size NEvents, true ~1/3 of the time and false ~2/3.
            setMask = (split==i)
            if debug:
                print(setTypes[i]+" mask shape:", setMask.shape)
                print("Time to make mask:", time.time()-setTime)

            setTime = time.time()
            print("Begin "+setTypes[i]+" split")

            # Create masked data
            # keyData is input data, by key. So (NEvents, BESvars) or (NEvents, FrameImages)
            # setMask has shape (NEvents,)
            # The ellipses are used for shape compability when masking. It interprets the dimensions needed.
            # setData is the masked data. So it should be ~1/3 of the size of keyData.
            setData = keyData[setMask,...]
            if debug: print("Time to copy data:", time.time()-setTime)

            setTime = time.time()
            print("Begin saving data")

            # Create dataset in output files.
            # The output files are {} with 3 keys: train, validation, test
            # besData is a {} with same three keys, each entry is also a {} whose keys are the 5 dataKeys: BESvars, 4xImageFrames
            # BESvars and ImageFrames have different shapes, hence the ifelse. Take shape from original input files.
            # There could be a better way to interpret the shapes automatically...
            if "frame" in thisKey.lower():
                besData[setTypes[i]][thisKey] = outputFiles[setTypes[i]].create_dataset(thisKey, data=setData, maxshape=(None, inputFile[thisKey].shape[1], inputFile[thisKey].shape[2], inputFile[thisKey].shape[3]), compression='lzf')
            else:
                besData[setTypes[i]][thisKey] = outputFiles[setTypes[i]].create_dataset(thisKey, data=setData, maxshape=(None, inputFile[thisKey].shape[1]), compression='lzf')
            if debug: print("Time to save data:", time.time()-setTime)
        
        if debug: print("Time to copy key", time.time()-keyTime)
        
    print("Finished splitting")
    if debug: print("Time to split file",time.time()-startTime)
    return 

# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) --all; 2) W,Z,b',
                        required=True)
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="~/nobackup/h5Dir/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default='~/nobackup/h5Dir/')
    parser.add_argument('-bs', '--batchSize',
                        type=int,
                        required=True)
    parser.add_argument('-d','--debug',
                        action='store_true')
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if args.debug:
        print("Samples to process: ", listOfSamples)

    # Check existance of directories you need
    if not os.path.isdir(args.h5Dir): 
        print(args.h5Dir, "does not exist")
        quit()
    if not os.path.isdir(args.outDir):
        os.mkdir(args.outDir)

    for sampleType in listOfSamples:
        print("Processing", sampleType)
        inputPath = args.h5Dir+sampleType+"Sample_BESTinputs.h5"
        #splitFileSlow(inputPath, args.outDir, args.debug)
        splitFileSKL(inputPath, args.outDir, args.debug, args.batchSize)
        
        
    ## Plot total pT distributions
    
    print("Done")

