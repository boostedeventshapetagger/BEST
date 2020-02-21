#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# test_boost_jetImageCreator.py ///////////////////////////////////////////////////
#==================================================================================
# This program makes boosted frame cosTheta phi jet images ////////////////////////
#==================================================================================

# modules
import sys
import ROOT as root
import uproot
import numpy
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import cProfile

# user modules
import tools.imageOperations as img

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

# set options 
plotJetImages = True
boostAxis = False
savePDF = False
savePNG = True 

# Start Profiling
#pr = cProfile.Profile()
#pr.enable()

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles and TTrees
upTree = uproot.open("../preprocess/preprocess_BEST_ZZ.root")["run/jetTree"]

# make file to store the images and BES variables
h5f = h5py.File("images/TestBoostedJetImages.h5","w")

# make a data frame to store the images
jetDF = {}

# make boosted jet images
#print "Starting with the Higgs Frame"
img.boostedJetPhotoshoot(upTree, "Higgs", 31, h5f, jetDF)

#==================================================================================
# Store BEST Variables ////////////////////////////////////////////////////////////
#==================================================================================

jetDF['BES_vars'] = upTree.pandas.df(["jetAK8_phi", "jetAK8_eta", "nSecondaryVertices", "jetAK8_Tau*",
                                       "FoxWolfram*",  "isotropy*", "aplanarity*", "thrust*", "subjet*mass*",
                                       "asymmetry*"])
                                       
print "show any NaNs", jetDF['BES_vars'].columns[jetDF['BES_vars'].isna().any()].tolist()

h5f.create_dataset('BES_vars', data=jetDF['BES_vars'], compression='lzf')
print "Stored Boosted Event Shape variables"

# disable profiling
#pr.disable()
#pr.print_stats(sort='time')

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
   print "Plotting Average Boosted jet images"
   img.plotAverageBoostedJetImage(jetDF['jet_images'], 'boost_Test', savePNG, savePDF)

   img.plotThreeBoostedJetImages(jetDF['jet_images'], 'boost_Test', savePNG, savePDF)

   img.plotMolleweideBoostedJetImage(jetDF['jet_images'], 'boost_Test', 31, savePNG, savePDF)

print "Mischief Managed!!!"

