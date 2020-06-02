import h5py
import argparse
import os
import numpy as np
import numpy.ma as ma
from sklearn.model_selection import train_test_split

listOfSamples = ["b","Higgs","QCD","Top","W","Z"]
listOfSampleTypess = ["","train","validation","test"]

def flattenFile(keepProbs, h5Dir, listOfSamples, myType, bins, binSize, maxRange, flattenIndex, userBatchSize):
    print("Begin flattening for", myType)
    for mySample in listOfSamples:
        filePath = h5Dir+mySample+'Sample_BESTinputs'
        if myType == "":
            filePath = filePath+".h5"
        else:
            filePath = filePath+"_"+myType+".h5"
        fIn = h5py.File(filePath, 'r')
        fOut = h5py.File(filePath.split('.')[0]+"_flattened.h5","w")
        besData = {}
        counter = 0
        totalEvents = fIn[fIn.keys()[0]].shape[0]
        print("Begin batching for sample", mySample, "total events", totalEvents)
        while (counter < totalEvents):
            batchSize = userBatchSize if (totalEvents > counter+userBatchSize) else (totalEvents-counter)
            print("Batch size", batchSize, "at", counter)
            myPtData = np.array(fIn["BES_vars"][counter:counter+batchSize,flattenIndex])
            print("Shape of myPtData", myPtData.shape)
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
                for myKey in fIn.keys():
                    print("Key", myKey)
                    myKeyData = np.array(fIn[myKey][counter:counter+batchSize,...])
                    print("Shape of myKeyData", myKeyData.shape)
                    result = myKeyData[myDataBool]
                    print("Shape of result", result.shape)
                    output = result
                    if myProbability < 1:
                        output = train_test_split(result, train_size=myProbability, shuffle=True)[0]
                    print("Size of kept events", len(output))
                    if not myKey in besData.keys():
                        print("Making new datset")
                        if "frame" in myKey.lower():
                            besData[myKey] = fOut.create_dataset(myKey, data=output, maxshape=(None, fIn[myKey].shape[1], fIn[myKey].shape[2], fIn[myKey].shape[3]), compression='lzf')
                        else:
                            besData[myKey] = fOut.create_dataset(myKey, data=output, maxshape=(None, fIn[myKey].shape[1]), compression='lzf')
                    else:
                        # append the dataset
                        print("Appending dataset")
                        besData[myKey].resize(besData[myKey].shape[0] + len(output[0]), axis=0)
                        besData[myKey][-len(output[0]) :] = output[0]
            counter += batchSize

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
        #print("Sample: ",listOfSamples[sampleIndex])
        #print("Bins: ", bins)
        #print("Probabilities", binnedProbs)

    #print(probs)
    return probs 

# Main function should take in arguments and call the functions you want
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
                        default=35)
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
                        default="/uscms/home/bonillaj/nobackup/h5samples/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default="/uscms/home/bonillaj/nobackup/h5samples/")
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
    if not os.path.isdir('plots'): os.mkdir('plots')
    if not os.path.isdir(args.outDir): os.mkdir(args.outDir)

    binSize = (args.rangeHigh-args.rangeLow)/args.nBins
    bins = [args.rangeLow+binSize*i for i in xrange(0,args.nBins)]
    if args.debug: print("Range: ", args.nBins," bins, from ", bins[0], " to ", bins[len(bins)-1]+binSize, " in steps of ", binSize) 
    if args.debug: print("Rejecting events above: ", args.rangeHigh)

    for myType in listOfSampleTypes:
        print("My Type", myType)
        keepProbs = getProbabilities(args.h5Dir, listOfSamples, myType, bins, binSize, args.rangeHigh, args.flattenIndex)
        flattenFile(keepProbs, args.h5Dir, listOfSamples, myType, bins, binSize, args.rangeHigh, args.flattenIndex, args.batchSize)

    #for mySample in listOfSamples:
     #   f = h5py.File('WSample_BESTinputs_test.h5', 'r')
#myData = f["BES_vars"]
#myDataPt = np.array(myData[...,35])
#myDataBool = myDataPt>1000
#mx = ma.masked_array(myDataPt, mask=myDataBool)
#myFinal = mx[~mx.mask]
