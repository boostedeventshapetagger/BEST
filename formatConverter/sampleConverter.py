#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# W_sample_formatConverter.py /////////////////////////////////////////////////////
#==================================================================================
# Author(s): Johan S Bonilla, Brendan Regnery -------------------------------------
# This program converts root ntuples to the python format necessar for training ///
# Inputs should be root files from preprocess
# Output should be three sets of hd5f files: trainingSet, validationSet, testignSet
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

# User modules
import tools.imageOperations as img

# Enter batch mode in root, so python can access displays
root.gROOT.SetBatch(True)

# Global variables
plotJetImages = True
listBESvars = False
savePDF = False
savePNG = True
stopAt = None
listOfSamples = ["b","Higgs","QCD","Top","W","Z"]
listOfFrameTypes = ["Higgs","Top","W","Z"]
treeName = "run/jetTree"
subNames = ["jet", "Jet", "nSecondaryVertices", "bDisc", "FoxWolf", 
            "isotropy_Higgs", "asymmetry", "aplanarity", "sphericity", "thrust"]

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================
def loadMC(eosDir, outDir, sampleType):
    # Find file paths that are relevant in eos and write the paths to a txt file
    # At the moment the user is expected to make the txt file by eosls dir >> listOfXfilePaths.txt
    # Remember the file paths should have root://cmseos.fnal.gov//eosDir
    # In the future this should be automated, hard part is working with eos from python
    # This file should (for now) live in your current directory (BEST/formatConverter/eosSamples/X.txt)

    # Store TFile paths
    print("Reading from", eosDir+"listOf"+sampleType+"filePaths.txt")
    with open(eosDir+"listOf"+sampleType+"filePaths.txt", 'r') as myFile:
        # Read file paths from txt file
        fileList = myFile.read().splitlines()
        print (fileList)
        
        # Make h5f file to store the images and BES variables
        h5fPath = outDir+sampleType+"Sample_BESTinputs.h5"
        print ("Writing h5f file to",h5fPath)
        h5f = h5py.File(h5fPath,"w")

        # Loop over input files
        numIter = 0
        besKeys = []
        imgFrames = {}
        besDS = None
        for arrays in uproot.iterate(fileList, treeName, entrysteps = 50000, namedecode='utf-8'):
            # get BES variable keys
            if numIter == 0:
                keys = arrays.keys()
                for key in keys :
                    for name in subNames :
                        if name in key and "candidate" not in key and "jetAK8_eta" not in key and "jetAK8_phi" not in key : 
                            if "px" not in key and "py" not in key and "pz" not in key and "energy" not in key : besKeys.append(key)
                print "There will be ", len(besKeys), " Input features stored"
                if listBESvars == True: print "Here are the stored BES vars ", besKeys
                if listBESvars == False: print "If you would like to list the BES vars, set listBESvars = True at the beginning of the code"
            (imgFrames, besDS) = storeBESTinputs(h5f, sampleType, numIter, arrays, besKeys, imgFrames, besDS)
            
            # if the stop iteration option is enabled
            if stopAt != None and stopAt <= besDS.shape[0] : 
                print "This program was told to stop early, please set 'stopAtIter = None' if you want it to run through all files"
                break
            
            # increment
            numIter += 1
            
    print("Finished loading MC")
    return 

#==================================================================================
# Store BEST Inputs ///////////////////////////////////////////////////////////////
#==================================================================================
#### This imgFrame thing needs to be nicer
def storeBESTinputs(h5f, sampleType, numIter, arrays, besKeys, imgFrames, besDS):
    # make a data frame to store the images and BES variables
    jetDF = {}

    # Store Frame Images
    for myType in listOfFrameTypes:
        jetDF[myType+'Frame_images'] = arrays[myType+'Frame_image']
        if numIter == 0:
            # make an h5 dataset
            imgFrames[myType] = h5f.create_dataset(myType+'Frame_images', data=jetDF[myType+'Frame_images'], maxshape=(None,31,31,1), compression='lzf')
        else:
            # append the dataset
            imgFrames[myType].resize(imgFrames[myType].shape[0] + jetDF[myType+'Frame_images'].shape[0], axis=0)
            imgFrames[myType][-jetDF[myType+'Frame_images'].shape[0] :] = jetDF[myType+'Frame_images'] 

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

    #==================================================================================
    # Plot Jet Images /////////////////////////////////////////////////////////////////
    #==================================================================================

    # Only plot jet Images for the first iteration (50,000)
    if plotJetImages == True and numIter == 0:
        print "Plotting Averaged images for the first 50,000 jets "
        for myFrameType in listOfFrameTypes:
            img.plotAverageBoostedJetImage(jetDF[myFrameType+'Frame_images'], sampleType+'Sample_'+myFrameType+'Frame', savePNG, savePDF)
        
        print "Plot the First 3 Images " 
        img.plotThreeBoostedJetImages(jetDF[myFrameType+'Frame_images'], sampleType+'Sample_'+myFrameType+'Frame', savePNG, savePDF)
    
    print("Finished storing BEST inputs")
    return (imgFrames, besDS)


# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) --all; 2) W,Z,b',
                        required=True)
    parser.add_argument('-t', '--training',
                        help="Type of training to prepare: batchGenerator or minShape",
                        dest='trainStyle',
                        required=True)
    parser.add_argument('-sa', '--stopAt',
                        type=int,
                        default=-1)
    parser.add_argument('-f', '--flatten',
                        action='store_true')
    parser.add_argument('-si','--standardInput',
                        action='store_true')
    parser.add_argument('-eos','--eosDir',
                        dest='eosDir',
                        default="eosSamples/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default="/uscms/home/bonillaj/nobackup/h5samples/")
    parser.add_argument('-d','--debug',
                        action='store_true')
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if args.stopAt > 0: stopAt = args.stopAt
    if args.debug:
        print("Samples to process: ", listOfSamples)
        print("Training Style: ", args.trainStyle)
        print("Flatten pT: ", args.flatten)
        print("Normalize features: ", args.standardInput)
        print("Reading Every nEvents: ", stopAt)

    # Make directories you need
    if not os.path.isdir('plots'): os.mkdir('plots')
    if not os.path.isdir('h5samples'): os.mkdir('h5samples')

    for sampleType in listOfSamples:
        print("Processing", sampleType)
        loadMC(args.eosDir, args.outDir, sampleType)
        
    ## Plot total pT distributions
    
    print("Done")




## Split into training, validation, and test sets
## Plot split pT distributions
## Make probabilities for flattening: for each bin, flatteningWeight=1 for least populous sample and flatteningWeight=nLeastSamp/nCurrentSamp otherwise

## Flatten pT spectrum
## Loop through each train, valid, test sets and record only those that pass flattening
## Output: 3 hd5f files (train, test, validate) -- Make sure to write to noBackup

## Normalize features in each of data sets (should be called 3 times)
## Output: 3 (smaller) hd5f files (train, test, validate) -- Make sure to write to noBackup




"""

print("Whoops")
exit()
# set options 
plotJetImages = True
listBESvars = False
savePDF = False
savePNG = True 
stopAt = None # this is for early stopping put 'None' if you want it to go through all files

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# Store TTree and TFile names
treeName = "run/jetTree"
wFile    = open("eosSamples/listOfWfilePaths.txt", 'r')
fileList = wFile.read().splitlines()

# make file to store the images and BES variables
h5f = h5py.File("h5samples/WSample_BESTinputs.h5","w")

# list to store the keys for BES variables
besKeys = []

# the BES variables that you would like to store
subNames = ["jet", "Jet", "nSecondaryVertices", "bDisc", "FoxWolf", 
            "isotropy_Higgs", "asymmetry", "aplanarity", "sphericity", "thrust"]

print "==================================================================================="
print "Welcome to the W Format Converter"
print "-----------------------------------------------------------------------------------"
print "This will convert the provided root files to an hdf5 python file"
print "-----------------------------------------------------------------------------------"

# Loop over input files
numIter = 0
for arrays in uproot.iterate(fileList, treeName, entrysteps = 50000, namedecode='utf-8'):

    #==================================================================================
    # Store BEST Inputs ///////////////////////////////////////////////////////////////
    #==================================================================================

    # get BES variable keys
    if numIter == 0:
        keys = arrays.keys()
        for key in keys :
            for name in subNames :
                if name in key and "candidate" not in key and "jetAK8_eta" not in key and "jetAK8_phi" not in key : 
                    if "px" not in key and "py" not in key and "pz" not in key and "energy" not in key : besKeys.append(key)
        print "There will be ", len(besKeys), " Input features stored"
        if listBESvars == True: print "Here are the stored BES vars ", besKeys
        if listBESvars == False: print "If you would like to list the BES vars, set listBESvars = True at the beginning of the code"
    

    # make a data frame to store the images and BES variables
    jetDF = {}

    # Store Higgs Frame Images
    jetDF['HiggsFrame_images'] = arrays['HiggsFrame_image']
    if numIter == 0:
        # make an h5 dataset
        imgHiggs = h5f.create_dataset('HiggsFrame_images', data=jetDF['HiggsFrame_images'], maxshape=(None,31,31,1), compression='lzf')
    else:
        # append the dataset
        imgHiggs.resize(imgHiggs.shape[0] + jetDF['HiggsFrame_images'].shape[0], axis=0)
        imgHiggs[-jetDF['HiggsFrame_images'].shape[0] :] = jetDF['HiggsFrame_images'] 

    # Store Top Frame Images
    jetDF['TopFrame_images'] = arrays['TopFrame_image']
    if numIter == 0:
        # make an h5 dataset
        imgTop = h5f.create_dataset('TopFrame_images', data=jetDF['TopFrame_images'], maxshape=(None,31,31,1), compression='lzf')
    else:
        # append the dataset
        imgTop.resize(imgTop.shape[0] + jetDF['TopFrame_images'].shape[0], axis=0)
        imgTop[-jetDF['TopFrame_images'].shape[0] :] = jetDF['TopFrame_images'] 

    # Store W Frame Images
    jetDF['WFrame_images'] = arrays['WFrame_image']
    if numIter == 0:
        # make an h5 dataset
        imgW = h5f.create_dataset('WFrame_images', data=jetDF['WFrame_images'], maxshape=(None,31,31,1), compression='lzf')
    else:
        # append the dataset
        imgW.resize(imgW.shape[0] + jetDF['WFrame_images'].shape[0], axis=0)
        imgW[-jetDF['WFrame_images'].shape[0] :] = jetDF['WFrame_images'] 

    # Store Z Frame Images
    jetDF['ZFrame_images'] = arrays['ZFrame_image']
    if numIter == 0:
        # make an h5 dataset
        imgZ = h5f.create_dataset('ZFrame_images', data=jetDF['ZFrame_images'], maxshape=(None,31,31,1), compression='lzf')
    else:
        # append the dataset
        imgZ.resize(imgZ.shape[0] + jetDF['ZFrame_images'].shape[0], axis=0)
        imgZ[-jetDF['ZFrame_images'].shape[0] :] = jetDF['ZFrame_images'] 

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

    #==================================================================================
    # Plot Jet Images /////////////////////////////////////////////////////////////////
    #==================================================================================

    # Only plot jet Images for the first iteration (50,000)
    if plotJetImages == True and numIter == 0:
        print "Plotting Averaged images for the first 50,000 jets "
        img.plotAverageBoostedJetImage(jetDF['HiggsFrame_images'], 'WSample_HiggsFrame', savePNG, savePDF)
        img.plotAverageBoostedJetImage(jetDF['TopFrame_images'], 'WSample_TopFrame', savePNG, savePDF)
        img.plotAverageBoostedJetImage(jetDF['WFrame_images'], 'WSample_WFrame', savePNG, savePDF)
        img.plotAverageBoostedJetImage(jetDF['ZFrame_images'], 'WSample_ZFrame', savePNG, savePDF)
        
        print "Plot the First 3 Images " 
        img.plotThreeBoostedJetImages(jetDF['HiggsFrame_images'], 'WSample_HiggsFrame', savePNG, savePDF)
        img.plotThreeBoostedJetImages(jetDF['TopFrame_images'], 'WSample_TopFrame', savePNG, savePDF)
        img.plotThreeBoostedJetImages(jetDF['WFrame_images'], 'WSample_WFrame', savePNG, savePDF)
        img.plotThreeBoostedJetImages(jetDF['ZFrame_images'], 'WSample_ZFrame', savePNG, savePDF)

    # increment
    numIter += 1

    # if the stop iteration option is enabled
    if stopAt != None and stopAt <= besDS.shape[0] : 
        print "This program was told to stop early, please set 'stopAtIter = None' if you want it to run through all files"
        break

print "Stored Boosted Event Shape Tagger Inputs"

print "Mischief Managed!!!"
"""
