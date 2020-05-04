#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MakeStandardInputs.py ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Author(s): Reyer Band ///////////////////////////////////////////////////////////
# This program makes Standardize Inputs ///////////////////////////////////////////
#----------------------------------------------------------------------------------

# modules
import numpy
import h5py
# get stuff from modules
from sklearn import preprocessing
from sklearn.externals import joblib
import json

#==================================================================================
# Load BES Vars /////////////////////////////////////////////////////////////////
#==================================================================================

# put BES variables in data frames
jetBESDF = {}
print "Getting QCD"
QCD = h5py.File("../formatConverter/h5samples/QCDSample_BESTinputs.h5","r")
jetBESDF['QCD'] = QCD['BES_vars'][()]
print type(jetBESDF['QCD'])
QCD.close()
print "Got QCD"
H = h5py.File("../formatConverter/h5samples/HiggsSample_BESTinputs.h5","r")
jetBESDF['H'] = H['BES_vars'][()]
H.close()
print "Got Higgs"
T = h5py.File("../formatConverter/h5samples/TopSample_BESTinputs.h5","r")
jetBESDF['T'] = T['BES_vars'][()]
T.close()
print "Got Top"
W = h5py.File("../formatConverter/h5samples/WSample_BESTinputs.h5","r")
jetBESDF['W'] = W['BES_vars'][()]
W.close()
print "Got W"
Z = h5py.File("../formatConverter/h5samples/ZSample_BESTinputs.h5","r")
jetBESDF['Z'] = Z['BES_vars'][()]
Z.close()
print "Got Z"
B = h5py.File("../formatConverter/h5samples/bSample_BESTinputs.h5","r")
jetBESDF['B'] = B['BES_vars'][()]
B.close()
print "Got B"

print "Accessed Jet Images and BES variables"

allBESinputs = numpy.concatenate([jetBESDF['QCD'], jetBESDF['H'], jetBESDF['T'], jetBESDF['W'], jetBESDF['Z'], jetBESDF['B']])
scaler = preprocessing.StandardScaler().fit(allBESinputs)

with open('ScalerParameters.txt', 'w') as outputFile:
   for mean,var in zip(scaler.mean_, scaler.var_):
      outputFile.write('{},{}\n'.format(mean, var))

for particle in ('QCD','H','T','W','Z','B'):
   jetBESDF[particle] = scaler.transform(jetBESDF[particle])
   print "Transformed", particle
   if particle != 'QCD':
      inf = h5py.File("images/"+particle+particle+".h5", "r")
   else:
      inf = h5py.File("images/"+particle+".h5", "r")
   outf = h5py.File("images/"+particle+"Transformed.h5", "w")
   print "Creating Dataset:", ""+particle+"Transformed.h5", len(jetBESDF[particle])
   outf.create_dataset(particle+'_BES', data=jetBESDF[particle], compression='lzf')
   #Copy the images to the new file
   #Treat QCD separately because of dumb labeling scheme I introduced
   if particle != 'QCD':
      outf.create_dataset(particle+'_H', data=inf[particle+particle+'_H_image'], compression='lzf')
      outf.create_dataset(particle+'_T', data=inf[particle+particle+'_T_image'], compression='lzf')
      outf.create_dataset(particle+'_W', data=inf[particle+particle+'_W_image'], compression='lzf')
      outf.create_dataset(particle+'_Z', data=inf[particle+particle+'_Z_image'], compression='lzf')
      outf.create_dataset(particle+'_LabFrameET', data=inf[particle+particle+'_LabFrameET'], compression='lzf')
   else:
      outf.create_dataset(particle+'_H', data=inf[particle+'_H_image'], compression='lzf')
      outf.create_dataset(particle+'_T', data=inf[particle+'_T_image'], compression='lzf')
      outf.create_dataset(particle+'_W', data=inf[particle+'_W_image'], compression='lzf')
      outf.create_dataset(particle+'_Z', data=inf[particle+'_Z_image'], compression='lzf')
      outf.create_dataset(particle+'_LabFrameET', data=inf[particle+'_LabFrameET'], compression='lzf')

   inf.close()
   outf.close()

