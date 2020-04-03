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
import ROOT as r
# get stuff from modules
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# set up keras


# Print which gpu/cpu this is running on

# set options 
savePDF = True
savePNG = True 
plotInputs = True
#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# Load images from h5 file
#h5f = h5py.File("images/phiCosThetaBoostedJetImages.h5","r")

# put images and BES variables in data frames
jetLabETDF = {}
QCD = h5py.File("images/QCD.h5","r")
jetLabETDF['QCD'] = QCD['QCD_LabFrameET'][()]
QCD.close()
H = h5py.File("images/HH.h5","r")
jetLabETDF['H'] = H['HH_LabFrameET'][()]
H.close()
T = h5py.File("images/tt.h5","r")
jetLabETDF['t'] = T['tt_LabFrameET'][()]
T.close()
W = h5py.File("images/WW.h5","r")
jetLabETDF['W'] = W['WW_LabFrameET'][()]
W.close()
Z = h5py.File("images/ZZ.h5","r")
jetLabETDF['Z'] = Z['ZZ_LabFrameET'][()]



B = h5py.File("images/BB.h5","r")
jetLabETDF['B'] = B['BB_LabFrameET'][()]
B.close()


print "Accessed Jet Images and BES variables"

w_dict = {}


batch_size = 200 #Amount of each class per batch, so really batch_size/6
pt_UpperBound = 1800
n_bins = 26
for label in ['QCD', 'H', 't', 'W', 'Z', 'B']:
   print label+':', len(jetLabETDF[label])
   w_dict[label] = numpy.zeros((len(jetLabETDF[label]), 1))
   it_hist = r.TH1F(label+'_source', label+'_source', n_bins, 500, pt_UpperBound)
   keep_list = []
   for entry, hist in enumerate(jetLabETDF[label]):
      it_hist.Fill(hist) #Literally just turning the numpy array in the h5 back into a root hist.
      pass
   it_flat = r.TH1F(label+'_flat', label+'_flat', n_bins, 500, pt_UpperBound)
   test_flat = r.TH1F(label+'_sel_flat', label+'_selected_indices', n_bins, 500, pt_UpperBound)

   for entry, hist in enumerate(jetLabETDF[label]):
      rand_num = numpy.random.uniform(0, 1)
#      print type(jetLabETDF[label]), type(hist), len(hist), type(hist[0])
      pt = hist
      if pt <= pt_UpperBound:
         keep_chance = 50 / float(it_hist.GetBinContent(it_hist.FindBin(pt)))
      else:
         keep_chance = 0
      w_dict[label][entry] = keep_chance
      if keep_chance > rand_num:
         it_flat.Fill(pt)
         keep_list.append(entry)
         pass
      pass
   numpy.random.shuffle(keep_list)
   print len(keep_list)
   keep_list = keep_list[0:batch_size]
   print len(keep_list)
   for index in keep_list:
      sel_pt = jetLabETDF[label][index]
      test_flat.Fill(sel_pt)
   canv = r.TCanvas('c1', 'c1')
   canv.cd()
   it_flat.Draw()
   it_flat.SetMinimum(0.0)
   canv.SaveAs('plots/Flat'+label+'_'+str(n_bins)+'bins.pdf')
   it_hist.Draw()
   it_hist.SetMinimum(0.0)
   canv.SaveAs('plots/Normal'+label+'_'+str(n_bins)+'bins.pdf')
   test_flat.Draw()
   test_flat.SetMinimum(0.0)
   canv.SaveAs('plots/SelectedIndices'+label+'_'+str(n_bins)+'bins'+str(batch_size)+'Batch.pdf')
   h5f = h5py.File('PtWeights/'+label+'EventWeights.h5', 'w')
   h5f.create_dataset(label, data=w_dict[label], compression='lzf')
   print label+' Number of Events:', it_flat.Integral()
   pass
