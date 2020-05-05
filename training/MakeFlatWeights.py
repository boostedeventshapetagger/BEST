#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MakeFlatWeights.py //////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Reyer Band ///////////////////////////////////////////////////////////
# This assigns weights for flattening pT //////////////////////////////////////////
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

# enter batch mode in root (so python can access displays)
r.gROOT.SetBatch(True)


# Print which gpu/cpu this is running on

# set options 
savePDF = True
savePNG = True 
plotInputs = True
#==================================================================================
# Load Jet Images /////////////////////////////////////////////////////////////////
#==================================================================================

# put images and BES variables in data frames
jetDF = {}
jetLabETDF = {}
QCD = h5py.File("../formatConverter/h5samples/QCDSample_BESTinputs.h5","r")
jetLabETDF['QCD'] = QCD['BES_vars'][...,28] # Make sure this index is correct if you change something!
QCD.close()
H = h5py.File("../formatConverter/h5samples/HiggsSample_BESTinputs.h5","r")
jetLabETDF['H'] = H['BES_vars'][...,28]
print jetLabETDF['H']
H.close()
T = h5py.File("../formatConverter/h5samples/TopSample_BESTinputs.h5","r")
jetLabETDF['t'] = T['BES_vars'][...,28]
T.close()
W = h5py.File("../formatConverter/h5samples/WSample_BESTinputs.h5","r")
jetLabETDF['W'] = W['BES_vars'][...,28]
W.close()
Z = h5py.File("../formatConverter/h5samples/ZSample_BESTinputs.h5","r")
jetLabETDF['Z'] = Z['BES_vars'][...,28]
Z.close()
B = h5py.File("../formatConverter/h5samples/bSample_BESTinputs.h5","r")
jetLabETDF['B'] = B['BES_vars'][...,28]
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

