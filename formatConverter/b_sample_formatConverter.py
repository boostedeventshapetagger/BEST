#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# b_sample_formatConverter.py /////////////////////////////////////////////////////
#==================================================================================
# Author(s): Brendan Regnery ------------------------------------------------------
# This program converts root ntuples to the python format necessar for training ///
#----------------------------------------------------------------------------------

# modules
import ROOT as root
import uproot
import numpy
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt

# user modules
import tools.imageOperations as img

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

# set options 
plotJetImages = True
listBESvars = False
savePDF = False
savePNG = True 

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles and TTrees
treeName = "run/jetTree"
fileList = ["/uscms/home/bonillaj/nobackup/samples/QCD/smallQCD.root"]

# make file to store the images and BES variables
h5f = h5py.File("h5samples/bSample_BESTinputs.h5","w")

# list to store the keys for BES variables
besKeys = []

# the BES variables that you would like to store
subNames = ["jet", "Jet", "nSecondaryVertices", "bDisc", "FoxWolf", 
            "isotropy_Higgs", "asymmetry", "aplanarity", "sphericity", "thrust"]

print "==================================================================================="
print "Welcome to the b Format Converter"
print "-----------------------------------------------------------------------------------"
print "This will convert the provided root files to an hdf5 python file"
print "-----------------------------------------------------------------------------------"

# Loop over input files
numIter = 0
for arrays in uproot.iterate(fileList, treeName, entrysteps = 50000, namedecode='utf-8'):
    print "Currently converting jets: ", numIter * 50000 + 1, " to ", (numIter + 1) * 50000

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
    h5f.create_dataset('HiggsFrame_images_' + str(numIter), data=jetDF['HiggsFrame_images'], compression='lzf')

    # Store Top Frame Images
    jetDF['TopFrame_images'] = arrays['TopFrame_image']
    h5f.create_dataset('TopFrame_images_' + str(numIter), data=jetDF['TopFrame_images'], compression='lzf')

    # Store W Frame Images
    jetDF['WFrame_images'] = arrays['WFrame_image']
    h5f.create_dataset('WFrame_images_' + str(numIter), data=jetDF['WFrame_images'], compression='lzf')

    # Store Z Frame Images
    jetDF['ZFrame_images'] = arrays['ZFrame_image']
    h5f.create_dataset('ZFrame_images_' + str(numIter), data=jetDF['ZFrame_images'], compression='lzf')

    # Store BES variables
    for key in besKeys :
        jetDF[key] = arrays[key]
        h5f.create_dataset('BES_vars_' + key + '_' + str(numIter), data=jetDF[key], compression='lzf')

    #==================================================================================
    # Plot Jet Images /////////////////////////////////////////////////////////////////
    #==================================================================================

    # Only plot jet Images for the first iteration (50,000)
    if plotJetImages == True and numIter == 0:
        print "Plotting Averaged images for the first 50,000 jets "
        img.plotAverageBoostedJetImage(jetDF['HiggsFrame_images'], 'bSample_HiggsFrame', savePNG, savePDF)
        img.plotAverageBoostedJetImage(jetDF['TopFrame_images'], 'bSample_TopFrame', savePNG, savePDF)
        img.plotAverageBoostedJetImage(jetDF['WFrame_images'], 'bSample_WFrame', savePNG, savePDF)
        img.plotAverageBoostedJetImage(jetDF['ZFrame_images'], 'bSample_ZFrame', savePNG, savePDF)
  
        print "Plot the First 3 Images " 
        img.plotThreeBoostedJetImages(jetDF['HiggsFrame_images'], 'bSample_HiggsFrame', savePNG, savePDF)
        img.plotThreeBoostedJetImages(jetDF['TopFrame_images'], 'bSample_TopFrame', savePNG, savePDF)
        img.plotThreeBoostedJetImages(jetDF['WFrame_images'], 'bSample_WFrame', savePNG, savePDF)
        img.plotThreeBoostedJetImages(jetDF['ZFrame_images'], 'bSample_ZFrame', savePNG, savePDF)

    # increment
    numIter += 1

print "Stored Boosted Event Shape Tagger Inputs"

print "Mischief Managed!!!"

