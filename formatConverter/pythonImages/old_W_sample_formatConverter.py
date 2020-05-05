#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# W_sample_formatConverter.py /////////////////////////////////////////////////////
#==================================================================================
# This program converts root ntuples to the python format necessar for training ///
#==================================================================================

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
boostAxis = False
savePDF = False
savePNG = True 

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles and TTrees
#upTree = uproot.open("/uscms_data/d3/bregnery/BEST/mc2017/preprocess_BEST_ZZ.root")["run/jetTree"]
upTree = uproot.open("../preprocess/BESTInputs_WSample.root")["run/jetTree"]

# make file to store the images and BES variables
h5f = h5py.File("h5samples/WSample_BESTinputs.h5","w")

# make a data frame to store the images and BES variables
jetDF = {}


#==================================================================================
# Store BEST Inputs ///////////////////////////////////////////////////////////////
#==================================================================================

# Store Higgs Frame Images
jetDF['HiggsFrame_images'] = upTree.arrays()[b'HiggsFrame_image']
h5f.create_dataset('HiggsFrame_images', data=jetDF['HiggsFrame_images'], compression='lzf')

# Store Top Frame Images
jetDF['TopFrame_images'] = upTree.arrays()[b'TopFrame_image']
h5f.create_dataset('TopFrame_images', data=jetDF['TopFrame_images'], compression='lzf')

# Store W Frame Images
jetDF['WFrame_images'] = upTree.arrays()[b'WFrame_image']
h5f.create_dataset('WFrame_images', data=jetDF['WFrame_images'], compression='lzf')

# Store Z Frame Images
jetDF['ZFrame_images'] = upTree.arrays()[b'ZFrame_image']
h5f.create_dataset('ZFrame_images', data=jetDF['ZFrame_images'], compression='lzf')

# Store BES variables
jetDF['BES_vars'] = upTree.pandas.df(["jetAK8*", "nSecondaryVertices",
                                       "FoxWolfram*", "aplanarity*", "thrust*", "subjet*mass" ])
h5f.create_dataset('BES_vars', data=jetDF['BES_vars'], compression='lzf')

print "Stored Boosted Event Shape variables"
print "show any NaNs", jetDF['BES_vars'].columns[jetDF['BES_vars'].isna().any()].tolist()

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
    print "Plotting Average Boosted jet images"
    img.plotAverageBoostedJetImage(jetDF['HiggsFrame_images'], 'WSample_HiggsFrame', savePNG, savePDF)
    img.plotAverageBoostedJetImage(jetDF['TopFrame_images'], 'WSample_TopFrame', savePNG, savePDF)
    img.plotAverageBoostedJetImage(jetDF['WFrame_images'], 'WSample_WFrame', savePNG, savePDF)
    img.plotAverageBoostedJetImage(jetDF['ZFrame_images'], 'WSample_ZFrame', savePNG, savePDF)
 
    img.plotThreeBoostedJetImages(jetDF['HiggsFrame_images'], 'WSample_HiggsFrame', savePNG, savePDF)
    img.plotThreeBoostedJetImages(jetDF['TopFrame_images'], 'WSample_TopFrame', savePNG, savePDF)
    img.plotThreeBoostedJetImages(jetDF['WFrame_images'], 'WSample_WFrame', savePNG, savePDF)
    img.plotThreeBoostedJetImages(jetDF['ZFrame_images'], 'WSample_ZFrame', savePNG, savePDF)

print "Mischief Managed!!!"

