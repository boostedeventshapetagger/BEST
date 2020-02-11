#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# boost_jetImageCreator.py ////////////////////////////////////////////////////////
#==================================================================================
# This program makes boosted frame cosTheta phi jet images ////////////////////////
#==================================================================================

# modules
import ROOT as root
import numpy
import pandas as pd
import h5py
#import matplotlib
#matplotlib.use('Agg') #prevents opening displays, must use before pyplot
#import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import copy
import random
import timeit
import sys

# user modules
#import functions as tools
#import imageOperations as img
import tools.functions as tools
import tools.imageOperations as img

from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras                                                                                                                                          

# enter batch mode in root (so python can access displays)
root.gROOT.SetBatch(True)

# set options 
plotJetImages = True
boostAxis = False
savePDF = True
savePNG = False

#Options related to smearing, irrelevant if smearImage is False
smearImage=False
smearWidth = 0.5
smearPoints= 50

filename = sys.argv[1]

#==================================================================================
# Load Monte Carlo ////////////////////////////////////////////////////////////////
#==================================================================================

# access the TFiles
#Storing data in /eos/, accessing through url
#fileCandidates = root.TFile.Open('root://cmsxrootd.fnal.gov//store/user/rband/UpdatedBESTSamples/'+filename)
fileCandidates = root.TFile.Open(filename)

filename = filename.replace('.root','')
# access the trees
treeCandidates = fileCandidates.Get("run/jetTree")
print treeCandidates, type(treeCandidates)

#Declare the file to be written to
#[:-5] strips .root from the filename
h5f = h5py.File("images/"+filename+".h5","w")
bestVars = tools.getBestBranchNames(treeCandidates)

# Loop over tree, making a numpy array for each image and the BES variables
print "Number of jets:", treeCandidates.GetEntries()
num_pass = 0
H_image = []
T_image = []
W_image = []
Z_image = []
BES_vars = []
ET = []
Mass = []

for index, jet in enumerate(treeCandidates):
   #Selection criteria hereB
   if index%1000 == 1: print "Imaging jet", index
   if (jet.jetAK8_pt > 500  and jet.jetAK8_SoftDropMass> 10 and jet.nSubjets_Higgs > 3): #nSubjets cut vetoes only ~0.1% of events, but many variables poorly defined if nSubjets < 4
      H_image.append(img.prepareBoostedImages(jet, 'H', 31, smearImage, smearWidth, smearPoints))
      T_image.append(img.prepareBoostedImages(jet, 'T', 31, smearImage, smearWidth, smearPoints))
      W_image.append(img.prepareBoostedImages(jet, 'W', 31, smearImage, smearWidth, smearPoints))
      Z_image.append(img.prepareBoostedImages(jet, 'Z', 31, smearImage, smearWidth, smearPoints))
      BES_vars.append(tools.GetBESVars(jet, bestVars))
      ET.append(jet.jetAK8_pt)
      Mass.append(jet.jetAK8_mass)
      num_pass += 1
      if num_pass%1000 == 1: print "Jet,", num_pass

h5f.create_dataset(filename+'_H_image', data=H_image, compression='lzf')
h5f.create_dataset(filename+'_T_image', data=T_image, compression='lzf')
h5f.create_dataset(filename+'_W_image', data=W_image, compression='lzf')
h5f.create_dataset(filename+'_Z_image', data=Z_image, compression='lzf')
h5f.create_dataset(filename+'_BES_vars', data=BES_vars, compression='lzf')
h5f.create_dataset(filename+'_LabFrameET', data=ET, compression='lzf')
h5f.create_dataset(filename+'_Mass', data=Mass, compression='lzf')

# plot with python
# if plotJetImages == True:
#    print "Plotting Average Boosted jet images"
#    img.plotAverageBoostedJetImage(jetImagesDF[filename+'_H_image'], filename+'boost_H', savePNG, savePDF)
#    img.plotThreeBoostedJetImages(jetImagesDF[filename+'_H_image'], filename+'boost_H', savePNG, savePDF)
#    img.plotAverageBoostedJetImage(jetImagesDF[filename+'_T_image'], filename+'boost_T', savePNG, savePDF)
#    img.plotThreeBoostedJetImages(jetImagesDF[filename+'_T_image'], filename+'boost_T', savePNG, savePDF)
#    img.plotAverageBoostedJetImage(jetImagesDF[filename+'_W_image'], filename+'boost_W', savePNG, savePDF)
#    img.plotThreeBoostedJetImages(jetImagesDF[filename+'_W_image'], filename+'boost_W', savePNG, savePDF)
#    img.plotAverageBoostedJetImage(jetImagesDF[filename+'_Z_image'], filename+'boost_Z', savePNG, savePDF)
#    img.plotThreeBoostedJetImages(jetImagesDF[filename+'_Z_image'], filename+'boost_Z', savePNG, savePDF)

print "Program was a great success!!!"

