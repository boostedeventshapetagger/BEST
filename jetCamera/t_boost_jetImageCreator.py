#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# top_boost_jetImageCreator.py ////////////////////////////////////////////////////
#==================================================================================
# This program makes boosted frame cosTheta phi jet images ////////////////////////
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
upTree = uproot.open("/uscms_data/d3/bregnery/BEST/mc2017/preprocess_BEST_tt.root")["run/jetTree"]

# make file to store the images and BES variables
h5f = h5py.File("images/topBoostedJetImages.h5","w")

# make a data frame to store the images
jetDF = {}

# make boosted jet images
print "Starting with the Higgs Frame"
img.boostedJetPhotoshoot(upTree, "Higgs", 31, h5f, jetDF)

#==================================================================================
# Store BEST Variables ////////////////////////////////////////////////////////////
#==================================================================================

jetDF['BES_vars'] = upTree.pandas.df(["jetAK8*", "nSecondaryVertices",
                                       "FoxWolfram*", "aplanarity*", "thrust*", "subjet*mass" ])

h5f.create_dataset('BES_vars', data=jetDF['BES_vars'], compression='lzf')
print "Stored Boosted Event Shape variables"

#==================================================================================
# Plot Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# plot with python
if plotJetImages == True:
   print "Plotting Average Boosted jet images"
   img.plotAverageBoostedJetImage(jetDF['jet_images'], 'boost_t', savePNG, savePDF)

   img.plotThreeBoostedJetImages(jetDF['jet_images'], 'boost_t', savePNG, savePDF)

   img.plotMolleweideBoostedJetImage(jetDF['jet_images'], 'boost_t', 31, savePNG, savePDF)

print "Mischief Managed!!!"

