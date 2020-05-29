#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# sampleSplitter.py /////////////////////////////////////////////////////
#==================================================================================
# Author(s): Johan S Bonilla, Brendan Regnery -------------------------------------
# This program splits h5f files into 3 smaller but equal orthogonal files       ///
# The point is to create separate train/validation/test samples of equal size   ///
# Inputs should be h5f files (can be flatted, or staright from formatConverter) ///
# Output should be three sets of hd5f files: training, validation, testign      ///
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

# User modules
import tools.imageOperations as img

# Enter batch mode in root, so python can access displays
root.gROOT.SetBatch(True)

# Global variables
listOfSampleTypes = ["b","Higgs","QCD","Top","W","Z"]

# Helper functions
def splitFile(inputPath, debug):
    listOfFrameTypes = ["Higgs","Top","W","Z"]
    setTypes = ["train", "validation", "test"]
    
    inputFile = h5py.File(inputPath,"r")
    dataKeys = inputFile.keys()
    totalEvents = inputFile[dataKeys[0]].shape[0]
    
    besData = {}
    outputFiles = {}
    for setType in setTypes:
        outputFiles[setType] = h5py.File(inputPath.split('.')[0]+"_"+setType+".h5","w")

    startTime = time.time()

    split = np.random.randint(3, size=totalEvents)

    besData[setTypes[0]] = {}
    for thisKey in dataKeys:
        keyTime = time.time()
        print("Spltting Key: "+thisKey)
        keyData = inputFile[thisKey]
        if debug: print("Key Data", keyData.shape)

        for i in range(0,3):
            setTime = time.time()
            print("Creating "+setTypes[i]+" set")
            setMask = (split==i)
            if debug:
                print(setTypes[i]+" mask shape:", setMask.shape)
                print("Time to make mask:", time.time()-keyTime)

            setTime = time.time()
            print("Begin "+setTypes[i]+" split")
            setData = keyData[setMask,...]
            if debug: print("Time to copy data:", time.time()-keyTime)

            setTime = time.time()
            print("Begin saving data")
            if "frame" in thisKey.lower():
                besData[setTypes[i]][thisKey] = outputFiles[setTypes[0]].create_dataset(thisKey, data=setData, maxshape=(None, inputFile[thisKey].shape[1], inputFile[thisKey].shape[2], inputFile[thisKey].shape[3]), compression='lzf')
            else:
                besData[setTypes[i]][thisKey] = outputFiles[setTypes[0]].create_dataset(thisKey, data=setData, maxshape=(None, inputFile[thisKey].shape[1]), compression='lzf')
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
                        default="h5samples/")
    parser.add_argument('-d','--debug',
                        action='store_true')
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if args.debug:
        print("Samples to process: ", listOfSampleTypes)

    # Make directories you need
    if not os.path.isdir(args.h5Dir): print(args.h5Dir, "does not exist")

    for sampleType in listOfSamples:
        print("Processing", sampleType)
        inputPath = args.h5Dir+sampleType+"Sample_BESTinputs.h5"
        splitFile(inputPath, args.debug)
        
        
    ## Plot total pT distributions
    
    print("Done")

