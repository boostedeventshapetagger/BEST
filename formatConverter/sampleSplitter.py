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
def storeBESinputs():
    # make a data frame to store the images and BES variables
    jetDF = {}

    # Store Frame Images
    for myType in listOfFrameTypes:
        jetDF[myType+'Frame_images'] = arrays[myType+'Frame_image']
        if numIter == 0:
            # make an h5 dataset
            imgFrame = h5f.create_dataset(myType+'Frame_images', data=jetDF[myType+'Frame_images'], maxshape=(None,31,31,1), compression='lzf')
        else:
            # append the dataset
            imgFrame.resize(imgFrame.shape[0] + jetDF[myType+'Frame_images'].shape[0], axis=0)
            imgFrame[-jetDF[myType+'Frame_images'].shape[0] :] = jetDF[myType+'Frame_images']

    # Store BES variables
    besList = []
    for key in besKeys :
        besList.append(arrays[key])
    jetDF['BES_vars'] = np.array(besList).T
    if numIter == 0:
        # make an h5 dataset
        besDS = h5f.create_dataset('BES_vars', data=jetDF['BES_vars'], maxshape=(None, len(besKeys)), compression='lzf')
    else:
        # append the dataset
        besDS.resize(besDS.shape[0] + len(jetDF['BES_vars']), axis=0)
        besDS[-len(jetDF['BES_vars']) :] = jetDF['BES_vars'] 

    print "Converted jets: ", besDS.shape[0] - len(jetDF['BES_vars']), " to ", besDS.shape[0]
    
    return

def storeImages():
    dset1 = f[u'BES_vars']
    return

def createRandomMasks(batchSize):
    randomList = [random.randint(0,2) for i in range(batchSize)]
    trainMask = [el==0 for el in randomList]
    validationMask = [el==1 for el in randomList]
    testMask = [el==2 for el in randomList]
    return (trainMask, validationMask, testMask)

def splitFile(inputPath, userBatchSize):
    listOfFrameTypes = ["Higgs","Top","W","Z"]
    setTypes = ["train", "validation", "test"]
    
    inputFile = h5py.File(inputPath,"r")
    dataKeys = inputFile.keys()
    totalEvents = inputFile[dataKeys[0]].shape[0]
    besData = {}
    outputFiles = {}
    outputFiles["train"] = h5py.File(inputPath.split('.')[0]+"_train.h5","w")
    outputFiles["validation"] = h5py.File(inputPath.split('.')[0]+"_validation.h5","w")
    outputFiles["test"] = h5py.File(inputPath.split('.')[0]+"_test.h5","w")
    
    counter = 0
    startTime = time.time()
    while (counter < totalEvents-1):
        whileTime = time.time()
        batchSize = userBatchSize if (totalEvents > counter+userBatchSize) else (totalEvents-counter)
        print("Batch size of ",batchSize,", at counter ",counter)
        randomMasks = createRandomMasks(batchSize)

        for i in range(0,3):
            setTime = time.time()
            print("Creating "+setTypes[i]+" set")
            if counter==0: besData[setTypes[i]] = {}
            #print("besKeys",besData.keys(),besData[setTypes[i]].keys())
            for thisKey in dataKeys:
                #print("besKeys",besData.keys(),besData[setTypes[i]].keys())
                keyTime = time.time()
                dsetMasked = [inputFile[thisKey][counter+j] for j in xrange(0,batchSize) if randomMasks[i][j]]
                print("Created masked dset for "+thisKey)
                if counter == 0:
                    #print("Making new DS of size",len(dsetMasked))
                    #print("Shape of inputFile", list(inputFile[thisKey].shape))
                    if "frame" in thisKey.lower():
                        besData[setTypes[i]][thisKey] = outputFiles[setTypes[i]].create_dataset(thisKey, data=dsetMasked, maxshape=(None, inputFile[thisKey].shape[1], inputFile[thisKey].shape[2], inputFile[thisKey].shape[3]), compression='lzf')
                        #print("besKeys",besData.keys(),besData[setTypes[i]].keys())
                    else:
                        besData[setTypes[i]][thisKey] = outputFiles[setTypes[i]].create_dataset(thisKey, data=dsetMasked, maxshape=(None, inputFile[thisKey].shape[1]), compression='lzf')
                        #print("besKeys",besData.keys(),besData[setTypes[i]].keys())
                else:
                    #print("Appending DS of size",len(dsetMasked))
                    #print("besKeys",besData.keys(),besData[setTypes[i]].keys())
                    #print(besData[setTypes[i]].keys())
                    besData[setTypes[i]][thisKey].resize(len(besData[setTypes[i]][thisKey]) + len(dsetMasked), axis=0)
                    besData[setTypes[i]][thisKey][-len(dsetMasked) :] = dsetMasked
                print("Time to copy key", time.time()-keyTime)
                #print("besKeys",besData.keys())
            print("Time to create set", time.time()-setTime)
        print("Time of batch", time.time()-whileTime)
        counter += batchSize
    print("Time to split file",time.time()-startTime)
        
    #for myFrame in listOfFrameTypes:
     #   imgFrame = h5f.create_dataset(myFrame+'Frame_images', data=jetDF[myFrame+'Frame_images'], maxshape=(None,31,31,1), compression='lzf')
      #  besData = h5f.create_dataset('BES_vars', data=[dsetTrain[i] for i in xrange(len(dsetTrain)) if msk[i]], maxshape=(None,94), compression='lzf')
            
    print("Finished splitting")
    return 

# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) --all; 2) W,Z,b',
                        required=True)
    parser.add_argument('-bs', '--batchSize',
                        type=int,
                        required=True)
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="h5samples/")
    parser.add_argument('-d','--debug',
                        action='store_true')
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if args.batchSize < 0: quit()
    if args.debug:
        print("Samples to process: ", listOfSampleTypes)
        print("Reading Every nEvents: ", args.batchSize)

    # Make directories you need
    #if not os.path.isdir('plots'): os.mkdir('plots')
    if not os.path.isdir(args.h5Dir): print(args.h5Dir, "does not exist")

    for sampleType in listOfSamples:
        print("Processing", sampleType)
        inputPath = args.h5Dir+sampleType+"Sample_BESTinputs.h5"
        splitFile(inputPath, args.batchSize)
        
        
    ## Plot total pT distributions
    
    print("Done")

