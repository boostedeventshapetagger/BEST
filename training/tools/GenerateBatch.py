import h5py
import numpy
import random
import keras
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot                                                                                                                          
import matplotlib.pyplot as plt
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.model_selection import train_test_split
from collections import OrderedDict
class GenerateBatch(object):
   def __init__(self, batch_size = 1200, validation_frac = 0.1, smearImage = False, debug_info = False, debug_plots = False):
      #File list should be a dict mapping the class to the file
      filelist = {'QCD' : 'images/QCDTransformed.h5', 'H' : 'images/HTransformed.h5', 't' : 'images/TTransformed.h5', 'W' : 'images/WTransformed.h5', 'Z' : 'images/ZTransformed.h5', 'B' : 'images/BTransformed.h5'}
      weightfilelist = {'QCD' : 'PtWeights/QCDEventWeights.h5', 'H' : 'PtWeights/HEventWeights.h5', 't' : 'PtWeights/tEventWeights.h5', 'W' : 'PtWeights/WEventWeights.h5', 'Z' : 'PtWeights/ZEventWeights.h5', 'B' : 'PtWeights/BEventWeights.h5'}
      self.filelist = filelist
      self.weightfilelist = weightfilelist
      self.batch_size = batch_size
      self.inputs = ['H', 'T', 'W', 'Z', 'BES']
#      self.inputs = ['H', 'T', 'BES']
      self.classes = ['QCD', 'H', 't', 'W', 'Z', 'B']
      self.num_classes = 6
      self.validation_frac = validation_frac
      self.debug_info = debug_info
      self.debug_plots = debug_plots
      self.smearImage = smearImage
      #Load all the data files into memory
      self.weights = self.OpenWeightFiles()
      self.data = self.OpenDataFiles()
      #Last batch saved only for debugging info
      self.last_train_keep = {'QCD' : [], 'H' : [], 'T' : [], 'W' : [], 'Z' : [], 'B' : []}
      self.last_valid_keep = {'QCD' : [], 'H' : [], 'T' : [], 'W' : [], 'Z' : [], 'B' : []}
      #Split the datasets in training and validation 
      self.train_indices, self.valid_indices =  self.split_train_valid()
      self.train_length = sum(len(x) for x in self.train_indices.values())
      self.valid_length = sum(len(x) for x in self.valid_indices.values())
      print ('Initialized Generator')

   def OpenWeightFiles(self):
      weight_dict = {}
      for flavor in self.classes:
         temp_file = h5py.File(self.weightfilelist[flavor], "r")
         weight_dict[flavor] = temp_file[flavor][()]
         temp_file.close()
      print ('Loaded All Weights Into Memory')
      return weight_dict

   def OpenDataFiles(self):
      data_dict = {}
      for flavor in self.classes:
         temp_file = h5py.File(self.filelist[flavor], "r")
         for label in self.inputs:
            data_dict[flavor+'_'+label] = temp_file[flavor+'_'+label][()]
         temp_file.close()
      print ('Loaded All Data Into Memory')
      if self.debug_plots:
         print (len(data_dict['QCD_BES']), len(data_dict['H_BES']), len(data_dict['t_BES']), len(data_dict['W_BES']), len(data_dict['Z_BES']), len(data_dict['B_BES']))
         print (len(data_dict['QCD_BES']), len(data_dict['QCD_BES'][0]))
         self.besInput_Labels =  ['jetAK8_pt', 'jetAK8_mass', 'jetAK8_SoftDropMass', 'nSecondaryVertices', 'bDisc', 'bDisc1', 'bDisc2', 'jetAK8_Tau4', 'jetAK8_Tau3', 'jetAK8_Tau2', 'jetAK8_Tau1', 'jetAK8_Tau32', 'jetAK8_Tau21', 'FoxWolfH1_Higgs', 'FoxWolfH2_Higgs', 'FoxWolfH3_Higgs', 'FoxWolfH4_Higgs', 'FoxWolfH1_Top', 'FoxWolfH2_Top', 'FoxWolfH3_Top', 'FoxWolfH4_Top', 'FoxWolfH1_W', 'FoxWolfH2_W', 'FoxWolfH3_W', 'FoxWolfH4_W', 'FoxWolfH1_Z', 'FoxWolfH2_Z', 'FoxWolfH3_Z', 'FoxWolfH4_Z', 'isotropy_Higgs', 'sphericity_Higgs', 'aplanarity_Higgs', 'thrust_Higgs', 'sphericity_Top', 'aplanarity_Top', 'thrust_Top', 'sphericity_W', 'aplanarity_W', 'thrust_W', 'sphericity_Z', 'aplanarity_Z', 'thrust_Z', 'njets_Higgs', 'njets_Top', 'njets_W', 'njets_Z', 'jet12_mass_Higgs', 'jet23_mass_Higgs', 'jet13_mass_Higgs', 'jet1234_mass_Higgs', 'jet12_mass_Top', 'jet23_mass_Top', 'jet13_mass_Top', 'jet1234_mass_Top', 'jet12_mass_W', 'jet23_mass_W', 'jet13_mass_W', 'jet1234_mass_W', 'jet12_mass_Z', 'jet23_mass_Z', 'jet13_mass_Z', 'jet1234_mass_Z', 'jet12_CosTheta_Higgs', 'jet23_CosTheta_Higgs', 'jet13_CosTheta_Higgs', 'jet1234_CosTheta_Higgs', 'jet12_CosTheta_Top', 'jet23_CosTheta_Top', 'jet13_CosTheta_Top', 'jet1234_CosTheta_Top', 'jet12_CosTheta_W', 'jet23_CosTheta_W', 'jet13_CosTheta_W', 'jet1234_CosTheta_W', 'jet12_CosTheta_Z', 'jet23_CosTheta_Z', 'jet13_CosTheta_Z', 'jet1234_CosTheta_Z', 'jet12_DeltaCosTheta_Higgs', 'jet13_DeltaCosTheta_Higgs', 'jet23_DeltaCosTheta_Higgs', 'jet12_DeltaCosTheta_Top', 'jet13_DeltaCosTheta_Top', 'jet23_DeltaCosTheta_Top', 'jet12_DeltaCosTheta_W', 'jet13_DeltaCosTheta_W', 'jet23_DeltaCosTheta_W', 'jet12_DeltaCosTheta_Z', 'jet13_DeltaCosTheta_Z', 'jet23_DeltaCosTheta_Z', 'asymmetry_Higgs', 'asymmetry_Top', 'asymmetry_W', 'asymmetry_Z']
         print(len(self.besInput_Labels))
         for i in range(0,len(data_dict['QCD_BES'][0])):
            if 'nSub' in self.besInput_Labels[i]: var_range = [3, 54]
            if 'mass' in self.besInput_Labels[i] or 'Mass' in self.besInput_Labels[i]: var_range = [0, 500]
            if 'pt' in self.besInput_Labels[i]: var_range = [0, 7000]
            plt.figure()
            plt.hist(data_dict['QCD_BES'][:,i], bins=50, color='b', label='QCD', histtype='step', normed=True)
            plt.hist(data_dict['H_BES'][:,i], bins=50, color='m', label='H', histtype='step', normed=True)
            plt.hist(data_dict['t_BES'][:,i], bins=50, color='r', label='t', histtype='step', normed=True)
            plt.hist(data_dict['W_BES'][:,i], bins=50, color='g', label='W', histtype='step', normed=True)
            plt.hist(data_dict['Z_BES'][:,i], bins=50, color='y', label='Z', histtype='step', normed=True)
            plt.hist(data_dict['B_BES'][:,i], bins=50, color='c', label='b', histtype='step', normed=True)
            plt.xlabel(self.besInput_Labels[i])
            plt.legend()
            plt.savefig("plots/Hist_"+self.besInput_Labels[i]+"_Scalarized.pdf")
            plt.close()
         print ('Plotted All BES Inputs')
      return data_dict

   def split_train_valid(self):
      keep_train_list = {}
      keep_valid_list = {}
      for flavor in self.classes:
         keep_train_list[flavor] = []
         keep_valid_list[flavor] = []
         pass
      #Each entry in this dict should be a list of indices, where the key is which class
      #Initialize a list with entries being indices of dataset, shuffle it to randomize order weights are viewed in.      
      #Loop over weights instead of dataset, much smaller amount to load                                                                                                                                           
      #This should give number per pT bin in weights                                                                                                                                                                 
      for flavor in self.classes:   
         temp_list = []
         #Save the indices to a list, and remove events with no chance of being kept
         for i in range(0, len(self.weights[flavor])):
            if self.weights[flavor][i] > 0:
               temp_list.append(i)

         #Now randomize those indices
         numpy.random.shuffle(temp_list)
#         train_stop_index = int((1-self.validation_frac) * len(self.weights[flavor]))
         train_stop_index = int((1-self.validation_frac) * len(temp_list))
         #Now assign a fraction to train, and another to list
         for index in temp_list[0:train_stop_index]:
            keep_train_list[flavor].append(index)

         for index in temp_list[train_stop_index:len(self.weights[flavor])]:
            keep_valid_list[flavor].append(index)
         print(flavor, len(keep_train_list[flavor]), len(keep_valid_list[flavor]))

      #If debugging, check how many events make it into each fraction, and keep track of intersections
      if self.debug_info:
         for flavor in self.classes:
            with open('TrainIndices_'+flavor+'.txt', 'w') as indexfile:
               for weight_val in keep_train_list[flavor]:
                  indexfile.write('%s \n' %weight_val)
               indexfile.close()
            with open('ValidIndices_'+flavor+'.txt', 'w') as indexfile:
               for weight_val in keep_valid_list[flavor]:
                  indexfile.write('%s \n' %weight_val)
               indexfile.close()
            set_train = set(keep_train_list[flavor])
            set_valid = set(keep_valid_list[flavor])
            print ('Are training/validation disjoint?', set_valid.isdisjoint(set_train))
            print(flavor, len(keep_train_list[flavor]), len(keep_valid_list[flavor]))
      return keep_train_list, keep_valid_list
   

   def ImageSmear(self, image, smearWidth = 0.5, npoints = 20):
      #Create empty image same size as input image, then populate with blur of original image
      nx = len(image)
      ny = len(image[0])
      blurOutput = numpy.zeros((nx, ny, 1))
      for x, column in enumerate(image):
         for y, entry in enumerate(column):
            if entry > 0:
               for n in xrange(0, npoints): # Upper limit here is number of points to spread the original into
                  gaussPointX = numpy.random.normal(x, smearWidth) #Width is in bin units - 0.5 means half a bin width, etc.
                  gaussPointY =  numpy.random.normal(y, smearWidth)
                  gaussPointX = gaussPointX % 31
                  gaussPointY = gaussPointY % 31
                  gaussPointX=round(gaussPointX)
                  gaussPointY=round(gaussPointY)
                  if gaussPointX == 31: gaussPointX = 0
                  if gaussPointY == 31: gaussPointY = 0
                  blurOutput[int(gaussPointX)][int(gaussPointY)] += float(entry)/float(npoints)

      return blurOutput

   def gaussSmear(self, batch, center = 0.01, width = 0.005):
      for x in xrange(0, len(batch)):
         for y in xrange(0, len(batch[x])):
            batch[x][y] += numpy.random.normal(center, width)
      return batch     

   #Method to generate training/validation data for one class, denoted as particle_key.  
   #Batch_type determines whether indices are fetched from the training list or validation list
   def generate_train_batch(self, particle_key, batch_type):
      keep_train_list = []
      
      if batch_type == 'train':
         for index in self.train_indices[particle_key]:
            rand_num = numpy.random.uniform(0, 1)
            weight = self.weights[particle_key][index]
            if weight > rand_num:
               keep_train_list.append(index)
         numpy.random.shuffle(keep_train_list)
         if len(keep_train_list) > (self.batch_size/self.num_classes):
            keep_train_list = keep_train_list[0:int(self.batch_size/self.num_classes)]
      elif batch_type == 'valid':
         for index in self.valid_indices[particle_key]:
            rand_num = numpy.random.uniform(0, 1)
            weight = self.weights[particle_key][index]
            if weight > rand_num:
               keep_train_list.append(index)
         numpy.random.shuffle(keep_train_list)
         if len(keep_train_list) > (self.batch_size/self.num_classes):
            keep_train_list = keep_train_list[0:int(self.batch_size/self.num_classes)]
         
      else:
         print ('Only train or valid are supported, this cant run')
         exit()


      if self.debug_info:
         if batch_type == "train":
            print ("Number of duplicated training events between batches: ", len(set(keep_train_list).intersection(set(self.last_train_keep[particle_key]))))
            print (set(keep_train_list).intersection(set(self.last_train_keep[particle_key])))
            self.last_train_keep[particle_key] = keep_train_list
         elif batch_type == "valid":
            print ("Number of duplicated validation events between batches: ", len(set(keep_train_list).intersection(set(self.last_valid_keep[particle_key]))))
            print (set(keep_train_list).intersection(set(self.last_valid_keep[particle_key])))
            self.last_valid_keep[particle_key] = keep_train_list


      return_train_batch = [] 
      temp_image_train_list = []
      best_vars_train_list = []

      if 'QCD' in particle_key: particle_index = 1
      if 'H' in particle_key: particle_index = 2
      if 't' in particle_key: particle_index = 3
      if 'W' in particle_key: particle_index = 4
      if 'Z' in particle_key: particle_index = 5
      if 'B' in particle_key: particle_index = 6

      for key in self.inputs:
         if 'BES' not in key:
            temp_image_train_list.append(numpy.zeros((len(keep_train_list), 31, 31, 1))) 


      for i, key in enumerate(self.inputs):
         if 'BES' not in key:
            for n, index in enumerate(keep_train_list):
               temp_image_train_list[i][n] = self.data[particle_key+'_'+key][index]
               if self.debug_info and n is 0 and i is 0:
                  print (key, type(self.data[particle_key+'_'+key][index]), len(self.data[particle_key+'_'+key][index]))


         if 'BES' in key:
            for n, index in enumerate(keep_train_list):
               best_vars_train_list.append(self.data[particle_key+'_'+key][index])
               if self.debug_info and n is 0:
                  print (key, type(self.data[particle_key+'_'+key][index]), len(self.data[particle_key+'_'+key][index]))

                  
#      print(len(best_vars_train_list), len(best_vars_train_list[0]), batch_type, particle_key)

      return_train_batch = [temp_image_train_list[0], temp_image_train_list[1], temp_image_train_list[2], temp_image_train_list[3], best_vars_train_list]
      
      if (self.smearImage):
         for i, key in enumerate(self.inputs):
            if 'BES' not in key:
               for m in xrange(0, len(return_train_batch[i])):
                  return_train_batch[i][m] = self.gaussSmear(return_train_batch[i][m])
      if self.debug_plots:
         print(len(best_vars_train_list), len(best_vars_train_list[0]), batch_type)
         for i in range(len(best_vars_train_list)):
            if 'nSub' in self.besInput_Labels[i]: var_range = [3, 54]
            if 'mass' in self.besInput_Labels[i] or 'Mass' in self.besInput_Labels[i]: var_range = [0, 500]
            if 'pt' in self.besInput_Labels[i]: var_range = [0, 7000]
            plt.figure()
            plt.hist(best_vars_train_list[i], bins=50, color='b', label=particle_key, histtype='step', normed=True)
            plt.xlabel(self.besInput_Labels[i])
            plt.legend()
            plt.savefig("plots/Hist_"+self.besInput_Labels[i]+"_Scalarized_Batch_"+particle_key+"_"+batch_type+".pdf")
            plt.close()

      return return_train_batch
   def train_looping(self, batch_type):
      #This uses generate_batch on each input file, returns the training tuples (data, truth)
      for index, particle in enumerate(self.classes):
         if 'QCD' in particle: particle_index = 0
         elif 'H' in particle: particle_index = 1
         elif 't' in particle: particle_index = 2
         elif 'W' in particle: particle_index = 3
         elif 'Z' in particle: particle_index = 4
         elif 'B' in particle: particle_index = 5

         train_temp = self.generate_train_batch(particle, batch_type)
         if index == 0:
            big_train_batch = train_temp
            big_train_truth = numpy.full(len(train_temp[0]), particle_index)
         else:
            big_train_truth = numpy.concatenate([big_train_truth, numpy.full(len(train_temp[0]), particle_index)])
            for i in range(0,len(train_temp)):
               big_train_batch[i] = numpy.concatenate([big_train_batch[i], train_temp[i]])
            if self.debug_info:
               print (index, particle, len(big_train_batch), len(big_train_batch[0]), len(big_train_batch[1]),  len(big_train_batch[2]),  len(big_train_batch[3]),  len(big_train_batch[4]))
               self.debug_info = False

      rng_state = numpy.random.get_state()
      numpy.random.shuffle(big_train_batch[0])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(big_train_batch[1])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(big_train_batch[2])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(big_train_batch[3])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(big_train_batch[4])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(big_train_truth)


      return big_train_batch, keras.utils.to_categorical(big_train_truth, num_classes = self.num_classes)

   def generator_train(self):
      while True:
         trainingDataset = self.train_looping('train')
         yield trainingDataset
   def generator_valid(self):
      while True:
         validDataset = self.train_looping('valid')
         yield validDataset


