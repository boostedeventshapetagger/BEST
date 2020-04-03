#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainBEST.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST: The Boosted Event Shape Tagger ////////////////////////
#==================================================================================

# modules
import numpy
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import copy
import random

# get stuff from modules
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


# user modules
import tools.functions as tools

# Print which gpu/cpu this is running on

# set options 
#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
#h5f = h5py.File("images/phiCosThetaBoostedJetImages.h5","r")

# put images and BES variables in data frames
jetImagesDF = {}
jetBESvarsDF = {}
jetLabETDF = {}
QCD = h5py.File("images/QCD.h5","r")
jetBESvarsDF['QCD'] = QCD['QCD_BES_vars'][()]
QCD.close()
H = h5py.File("images/HH.h5","r")
jetBESvarsDF['H'] = H['HH_BES_vars'][()]
H.close()
T = h5py.File("images/tt.h5","r")
jetBESvarsDF['t'] = T['tt_BES_vars'][()]
T.close()
W = h5py.File("images/WW.h5","r")
jetBESvarsDF['W'] = W['WW_BES_vars'][()]
W.close()
Z = h5py.File("images/ZZ.h5","r")
jetBESvarsDF['Z'] = Z['ZZ_BES_vars'][()]
Z.close()
B = h5py.File("images/BB.h5","r")
jetBESvarsDF['B'] = B['BB_BES_vars'][()]
B.close()




print "Made image dataframes"

#==================================================================================
# Train the Neural Network ////////////////////////////////////////////////////////
#==================================================================================

# Store data and truth

var_names =  ['jetAK8_pt', 'jetAK8_mass', 'jetAK8_SoftDropMass', 'nSecondaryVertices', 'bDisc', 'bDisc1', 'bDisc2', 'jetAK8_Tau4', 'jetAK8_Tau3', 'jetAK8_Tau2', 'jetAK8_Tau1', 'jetAK8_Tau32', 'jetAK8_Tau21', 'FoxWolfH1_Higgs', 'FoxWolfH2_Higgs', 'FoxWolfH3_Higgs', 'FoxWolfH4_Higgs', 'FoxWolfH1_Top', 'FoxWolfH2_Top', 'FoxWolfH3_Top', 'FoxWolfH4_Top', 'FoxWolfH1_W', 'FoxWolfH2_W', 'FoxWolfH3_W', 'FoxWolfH4_W', 'FoxWolfH1_Z', 'FoxWolfH2_Z', 'FoxWolfH3_Z', 'FoxWolfH4_Z', 'isotropy_Higgs', 'sphericity_Higgs', 'aplanarity_Higgs', 'thrust_Higgs', 'sphericity_Top', 'aplanarity_Top', 'thrust_Top', 'sphericity_W', 'aplanarity_W', 'thrust_W', 'sphericity_Z', 'aplanarity_Z', 'thrust_Z', 'nSubjets_Higgs', 'nSubjets_Top', 'nSubjets_W', 'nSubjets_Z', 'subjet12_mass_Higgs', 'subjet23_mass_Higgs', 'subjet13_mass_Higgs', 'subjet1234_mass_Higgs', 'subjet12_mass_Top', 'subjet23_mass_Top', 'subjet13_mass_Top', 'subjet1234_mass_Top', 'subjet12_mass_W', 'subjet23_mass_W', 'subjet13_mass_W', 'subjet1234_mass_W', 'subjet12_mass_Z', 'subjet23_mass_Z', 'subjet13_mass_Z', 'subjet1234_mass_Z', 'subjet12_CosTheta_Higgs', 'subjet23_CosTheta_Higgs', 'subjet13_CosTheta_Higgs', 'subjet1234_CosTheta_Higgs', 'subjet12_CosTheta_Top', 'subjet23_CosTheta_Top', 'subjet13_CosTheta_Top', 'subjet1234_CosTheta_Top', 'subjet12_CosTheta_W', 'subjet23_CosTheta_W', 'subjet13_CosTheta_W', 'subjet1234_CosTheta_W', 'subjet12_CosTheta_Z', 'subjet23_CosTheta_Z', 'subjet13_CosTheta_Z', 'subjet1234_CosTheta_Z', 'subjet12_DeltaCosTheta_Higgs', 'subjet13_DeltaCosTheta_Higgs', 'subjet23_DeltaCosTheta_Higgs', 'subjet12_DeltaCosTheta_Top', 'subjet13_DeltaCosTheta_Top', 'subjet23_DeltaCosTheta_Top', 'subjet12_DeltaCosTheta_W', 'subjet13_DeltaCosTheta_W', 'subjet23_DeltaCosTheta_W', 'subjet12_DeltaCosTheta_Z', 'subjet13_DeltaCosTheta_Z', 'subjet23_DeltaCosTheta_Z', 'asymmetry_Higgs', 'asymmetry_Top', 'asymmetry_W', 'asymmetry_Z']
print len(var_names)
print len(jetBESvarsDF['QCD'][0]), len(jetBESvarsDF['QCD'][:]), len(jetBESvarsDF['QCD'][:,0])
print(type(jetBESvarsDF['QCD']), type(jetBESvarsDF['QCD'][0]), type(jetBESvarsDF['QCD'][0][0]))

for i in range(0,len(jetBESvarsDF['QCD'][0])):
    var_range = [-3,3]
    if 'nSub' in var_names[i]: var_range = [3, 54]
    if 'mass' in var_names[i] or 'Mass' in var_names[i]: var_range = [0, 500]
    if 'pt' in var_names[i]: var_range = [0, 7000]
    plt.figure()
    plt.hist(jetBESvarsDF['QCD'][:,i], bins=51, range=var_range, color='b', label='QCD', histtype='step', normed=True)
    plt.hist(jetBESvarsDF['H'][:,i], bins=51, range=var_range, color='m', label='H', histtype='step', normed=True)
    plt.hist(jetBESvarsDF['t'][:,i], bins=51, range=var_range, color='r', label='t', histtype='step', normed=True)
    plt.hist(jetBESvarsDF['W'][:,i], bins=51, range=var_range, color='g', label='W', histtype='step', normed=True)
    plt.hist(jetBESvarsDF['Z'][:,i], bins=51, range=var_range, color='y', label='Z', histtype='step', normed=True)
    plt.hist(jetBESvarsDF['B'][:,i], bins=51, range=var_range, color='c', label='B', histtype='step', normed=True)
    plt.xlabel(var_names[i])
    plt.legend()
    plt.savefig("plots/Hist_"+var_names[i]+".pdf")
    plt.close()


