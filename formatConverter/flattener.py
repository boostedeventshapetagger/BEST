#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# flattener.py /////////////////////////////////////////////////////
#==================================================================================
# Author(s): Johan S Bonilla, Brendan Regnery, Reyer Band -------------------------
# This program takes in h5 files and equalizes the number of events based on pt ///
# Inputs should be h5 files after splitting the samples
# The process plots the various files and, bin by bin, keeps all events of the
# least populous sample, and throws away events based on probabilty ratio
# P=NumEventsMinSample/NumEventsThisSample
# Output are h5 samples.
#----------------------------------------------------------------------------------

import h5py
import argparse
import os
import numpy as np
import numpy.ma as ma
from sklearn.model_selection import train_test_split

listOfSamples = ["b","Higgs","QCD","Top","W","Z"]
listOfSampleTypess = ["","train","validation","test"]

## Loop over all files and keep/reject events in batches. One uses train_test_split to do the heavy lifting.
## The random state in the function ensures the same mask across the keys. Since the probabilities are binned in pt,
## one must loop over all pt bins and evaluate keep/reject for each.
def flattenFile(keepProbs, h5Dir, outDir, listOfSamples, myType, bins, binSize, maxRange, flattenIndex, userBatchSize):
    print("Begin flattening for", myType)
    
    ## Loop over all samples and flatten each one by one. Probabilities have been previously calculated and passed in
    for mySample in listOfSamples:
        filePath = h5Dir+mySample+'Sample_BESTinputs'
        if myType == "":
            filePath = filePath+".h5"
        else:
            filePath = filePath+"_"+myType+".h5"
        fIn = h5py.File(filePath, 'r')
        fOut = h5py.File(outDir+filePath.split('.')[-2].split('/')[-1]+"_flattened.h5","w")
        besData = {}
        counter = 0
        totalEvents = fIn[fIn.keys()[0]].shape[0]
        print("Begin batching for sample", mySample, "total events", totalEvents)

        ## Perform the keep/throw-away opration in batches. Counter eventually reaches end of numEvents in file. Incrementer at end of while loop
        while (counter < totalEvents):
            batchSize = userBatchSize if (totalEvents > counter+userBatchSize) else (totalEvents-counter)
            print("Batch size", batchSize, "at", counter)

            # Grab pt part of dataset to evaluate whether to keep or reject event
            myPtData = np.array(fIn["BES_vars"][counter:counter+batchSize,flattenIndex])
            print("Shape of myPtData", myPtData.shape)

            # For each key in dataset, take partial dataset and copy or reject. Note the random state below ensures the same set of events are kept or rejected 
            for myKey in fIn.keys():
                print("Key", myKey)
                myKeyData = np.array(fIn[myKey][counter:counter+batchSize,...])
                print("Shape of myKeyData", myKeyData.shape)
                # Loop over bins (events in dataset may belong to any pt-bin)
                for binIndex in range(0,len(bins)):
                    myProbability = keepProbs[listOfSamples.index(mySample)][binIndex]
                    if myProbability == 0:
                        print("Probability is 0, skipping saving part")
                        continue
                    currLowRange = bins[binIndex]
                    currHighRange = min(currLowRange+binSize, maxRange)
                    ## Pick out data in bin, myDataBool has shape (batchSize,1) with boolean values of whether the event is in the right bin
                    myDataBool = (currLowRange<myPtData)*(myPtData<currHighRange)
                    print("Processing Bin:", currLowRange, currHighRange)
                    print("Shape of myDataBool", myDataBool.shape)
                    result = myKeyData[myDataBool]
                    print("Shape of result", result.shape)
                    if result.shape[0] == 0:
                        print("Result has no events in bin, continue to next bin")
                        continue
                    output = result
                    if myProbability < 1:
                        ## The random state needs to be the same for each key to ensure we keep the same events across keys
                        output = train_test_split(result, train_size=myProbability, shuffle=True, random_state=29)[0]
                    print("Size of kept events", len(output))
                    if len(output) == 0:
                        print("Output has no events in bin, continue to next bin")
                        continue
                    # Store kept data
                    if not myKey in besData.keys():
                        print("Making new datset")
                        if "frame" in myKey.lower():
                            besData[myKey] = fOut.create_dataset(myKey, data=output, maxshape=(None, fIn[myKey].shape[1], fIn[myKey].shape[2], fIn[myKey].shape[3]), compression='lzf')
                        else:
                            besData[myKey] = fOut.create_dataset(myKey, data=output, maxshape=(None, fIn[myKey].shape[1]), compression='lzf')
                    else:
                        # append the dataset
                        print("Appending dataset")
                        besData[myKey].resize(besData[myKey].shape[0] + len(output), axis=0)
                        besData[myKey][-len(output) :] = output
            counter += batchSize

## Plot samples in pt (or variable of choice) and return a list of probabilities for keeping events
def getProbabilities(h5Dir, listOfSamples, myType, bins, binSize, maxRange, flattenIndex):
    print("Begin making probabilities array")
    probs = [] # First axis is listOfSamples, second axis is ptBins, values are probability to keep event in sample,ptBin
    binnedNEvents = [] # First axis is listOfSamples, second axis is ptBins, values are number of events in sample,ptBin

    ## The following block should populate the binnedNEvents list
    for mySample in listOfSamples:
        print("Processing", mySample)
        filePath = h5Dir+mySample+'Sample_BESTinputs'
        if myType == "":
            filePath = filePath+".h5"
        else:
            filePath = filePath+"_"+myType+".h5"
        f = h5py.File(filePath, 'r')
        binnedNEvents.append([])

        ## Only needs to be done on smallest key, BEST_vars
        ## Output shape of myData is (NEvents,)
        myData = np.array(f["BES_vars"][...,flattenIndex])
        print("Begin bin looping")
        for currLowRange in bins:
            currHighRange = min(currLowRange+binSize, maxRange)
            ## Pick out data in bin
            myDataBool = (currLowRange<myData)*(myData<currHighRange)
            ## (bool = True -> mask) so need to invert mask to keep desired info
            dataMask = ma.masked_array(myData, mask=~myDataBool)
            ## Invert mask again (Maybe this could be cleaner)
            ## Shape of truncated data is (NEventsPass,)
            myTruncatedData = dataMask[~dataMask.mask]
            ## Append NEvents in bin to last element (list) of binnedNEvents, i.e. mySample
            binnedNEvents[-1].append(len(myTruncatedData))
    #print(binnedNEvents)

    ## Convert to numpy array to better manipulate
    ## binnedNEvents is shape (nSamples, nBins, 1) with values NEventsInBinForSample
    binnedNEvents = np.array(binnedNEvents)

    ## Next, populate probs which is a list of shape (NSamples, NBins, 1) with value keepProbability
    print("Begin making prob calculations")
    for sampleIndex in range(0, len(listOfSamples)):
        binnedProbs = []
        for binIndex in range(0, len(bins)):
            num = float(min(binnedNEvents[...,binIndex]))
            denom = float(binnedNEvents[sampleIndex][binIndex])
            if denom > 0:
                binnedProbs.append(num/denom)
            else:
                binnedProbs.append(0.)
        probs.append(binnedProbs)

    return probs 

# Main function should take in arguments and call the functions you want
# -s is the samples to process: if 'all' then it does QCD,W,Z,Top,b,Higgs. Else you can provide a comma separated list
# -st is the types of sample sets to process, i.e. train, validation, test. If 'all' then it does these three but also the pre-split samples
# -b is the batch size to do the copying when flattening. This is a performance hyper-parameter. The output is unaffected.
# -fi flattenIndex is the BESvars index to flatten on. Currenltly the default is 28 since that corresponds to pt in the current samples.
# -rl rangeLow is the lower limit to set the bins. Anything below this will always be rejected.
# -rh rangeHigh is the upper limit to set the bins. Anything above this will always be rejected.
# -nb is the number of bins for the flattening range. Bin size is set by (rl-rh)/nbins.
# -hd is the path to the location of the input h5 files
# -o is the path to the location where to send the outputs
# -d enables the debugging flag with all print statements
if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) all; 2) W,Z,b',
                        required=True)
    parser.add_argument('-st', '--sampleTypes',
                        dest='sampleTypes',
                        help='<Required> Which (comma separated) sample types to process. Examples: 1) --all (includes pre-split); 2) train,validation,test',
                        required=True)
    parser.add_argument('-b', '--batchSize',
                        dest='batchSize',
                        type=int,
                        default=-1)
    parser.add_argument('-fi', '--flattenIndex',
                        dest='flattenIndex',
                        type=int,
                        default=28)
    parser.add_argument('-rl', '--rangeLow',
                        dest='rangeLow',
                        type=float,
                        default=0)
    parser.add_argument('-rh', '--rangeHigh',
                        dest='rangeHigh',
                        type=float,
                        default=3500)
    parser.add_argument('-nb', '--nBins',
                        dest='nBins',
                        type=int,
                        default=175)
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="~/nobackup/h5samples/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default="~/nobackup/h5samples/")
    parser.add_argument('-d','--debug',
                        action='store_true')
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if not args.sampleTypes == "all": listOfSampleTypes = args.sampleTypes.split(',')
    if args.debug:
        print("Samples to process: ", listOfSamples)
        print("Flattenning Index: ", args.flattenIndex)
        print("Reading Every nEvents: ", args.batchSize)

    # Make directories you need
    if not os.path.isdir(args.outDir): os.mkdir(args.outDir)

    binSize = (args.rangeHigh-args.rangeLow)/args.nBins
    bins = [args.rangeLow+binSize*i for i in xrange(0,args.nBins)]
    if args.debug: print("Range: ", args.nBins," bins, from ", bins[0], " to ", bins[len(bins)-1]+binSize, " in steps of ", binSize) 
    if args.debug: print("Rejecting events above: ", args.rangeHigh)

    for myType in listOfSampleTypes:
        print("My Type", myType)
        keepProbs = getProbabilities(args.h5Dir, listOfSamples, myType, bins, binSize, args.rangeHigh, args.flattenIndex)
        flattenFile(keepProbs, args.h5Dir, args.outDir, listOfSamples, myType, bins, binSize, args.rangeHigh, args.flattenIndex, args.batchSize)

