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
# Convert /////////////////////////////////////////////////////////////////////////
#==================================================================================
def convert(eosDir, outDir, sampleType, debug):
    # Find file paths that are relevant in eos and write the paths to a txt file
    # At the moment the user is expected to make the txt file by eosls dir >> listOfXfilePaths.txt
    # If the files are on eos, their paths should have 'root://cmseos.fnal.gov//eosDir' as a prefix
    # In the future this should be automated, hard part is working with eos from python...
    # This file should (for now) live in your current directory (BEST/formatConverter/eosSamples/listOf<sampleType>filePath.txt)

    # Open file and read lines (individual file paths)
    if debug: print("Reading from", eosDir+"listOf"+sampleType+"filePaths.txt")
    with open(eosDir+"listOf"+sampleType+"filePaths.txt", 'r') as myFile:
        # Read file paths from txt file
        fileList = myFile.read().splitlines()
        if debug: print (fileList)
        
        # Make h5f output file to store the images and BES variables
        h5fPath = outDir+sampleType+"Sample_BESTinputs.h5"
        if debug: print ("Writing h5f file to",h5fPath)
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
                if debug: print("There will be ", len(besKeys), " Input features stored")
                if listBESvars == True: print("Here are the stored BES vars ", besKeys)
                if listBESvars == False: print("If you would like to list the BES vars, set listBESvars = True at the beginning of the code")
            (imgFrames, besDS) = storeBESTinputs(h5f, sampleType, numIter, arrays, besKeys, imgFrames, besDS, debug)
            
            # if the stop iteration option is enabled
            if stopAt != None and stopAt <= besDS.shape[0] : 
                print("This program was told to stop early, please set 'stopAtIter = None' if you want it to run through all files")
                break
            
            # increment
            numIter += 1
            
    print("Finished loading MC")
    return 

#==================================================================================
# Store BEST Inputs ///////////////////////////////////////////////////////////////
#==================================================================================
#### This imgFrame thing needs to be nicer
def storeBESTinputs(h5f, sampleType, numIter, arrays, besKeys, imgFrames, besDS, debug):
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

    if debug:  
        print("Converted jets: ", besDS.shape[0] - len(jetDF['BES_vars']), " to ", besDS.shape[0])

    #==================================================================================
    # Plot Jet Images /////////////////////////////////////////////////////////////////
    #==================================================================================

    # Only plot jet Images for the first iteration (50,000)
    if plotJetImages == True and numIter == 0:
        if debug: print("Plotting Averaged images for the first 50,000 jets")
        for myFrameType in listOfFrameTypes:
            img.plotAverageBoostedJetImage(jetDF[myFrameType+'Frame_images'], sampleType+'Sample_'+myFrameType+'Frame', savePNG, savePDF)
        
        if debug: print("Plot the First 3 Images") 
        img.plotThreeBoostedJetImages(jetDF[myFrameType+'Frame_images'], sampleType+'Sample_'+myFrameType+'Frame', savePNG, savePDF)
    
    if debug: print("Finished storing BEST inputs")
    return (imgFrames, besDS)


## Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    
    # Take in arguments and set driving variables
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) --all; 2) W,Z,b',
                        required=True)
    parser.add_argument('-sa', '--stopAt',
                        type=int,
                        default=-1)
    parser.add_argument('-p', '--plots',
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

    # Diagnostic debug
    if args.debug:
        print("Samples to process:", listOfSamples)
        print("Reading every nEvents:", stopAt)
        print("Generate diagnostic plots:", args.plots)

    # Make directories you need
    if not os.path.isdir('plots'): os.mkdir('plots')
    if not os.path.isdir('h5samples'): os.mkdir('h5samples')

    # Loop over samples and convert each separately
    for sampleType in listOfSamples:
        print("Processing", sampleType)
        convert(args.eosDir, args.outDir, sampleType, args.debug)
        
    ## Plot total pT distributions
    #if args.plots:
    #    callPlotter
    
    print("Done")
