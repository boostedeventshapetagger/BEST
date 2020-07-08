#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MakeStandardInputs.py ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Author(s): Reyer Band, Johan S. Bonilla, Brendan Regnary ////////////////////////
# This program makes Standardize Inputs ///////////////////////////////////////////
#----------------------------------------------------------------------------------

# modules
import numpy
import h5py
# get stuff from modules
from sklearn import preprocessing
from sklearn.externals import joblib
import json
import argparse, os

listOfSamples = ["b","Higgs","QCD","Top","W","Z"]
setTypes = ["","train","validation","test"]


#==================================================================================
# Standardize BES Vars /////////////////////////////////////////////////////////////////
#==================================================================================
def standardizeBESTVars(fileDir = "../formatConverter/h5samples/", sampleTypes = ["QCD","Higgs","Top","W","Z","b"], setTypes = [""], suffix = ""):
   # put BES variables in data frames
   for mySet in setTypes:
      jetBESDF = {}
      for mySample in sampleTypes:
         print("Getting", mySample, mySet)
         filePath = fileDir+mySample+"Sample_BESTinputs"
         if not mySet == "":
            filePath = filePath + "_" + mySet
         if suffix == "":
            filePath = filePath + ".h5"
         else:
            filePath = filePath + "_" + suffix + ".h5"
         myF = h5py.File(filePath,"r")
         jetBESDF[mySample] = myF['BES_vars'][()]
         print(type(jetBESDF[mySample]), jetBESDF[mySample].shape)
         myF.close()
         print("Got", mySample, mySet)

      print("Accessed BES variables for", mySet)

      allBESinputs = numpy.concatenate([jetBESDF[mySample] for mySample in sampleTypes])
      print("Shape allBESinputs", allBESinputs.shape)
      scaler = preprocessing.StandardScaler().fit(allBESinputs)

      with open('ScalerParameters_'+mySet+'.txt', 'w') as outputFile:
         for mean,var in zip(scaler.mean_, scaler.var_):
            outputFile.write('{},{}\n'.format(mean, var))

      print("JetBESDF", jetBESDF.keys())
      for mySample in sampleTypes:
         jetBESDF[mySample] = scaler.transform(jetBESDF[mySample])
         print("Transformed", mySample)
         #if infParticle == 'H' : infParticle = 'Higgs'
         #if infParticle == 'T' : infParticle = 'Top'
         #if infParticle == 'B' : infParticle = 'b'
         outFilePath = fileDir+mySample+"Sample_BESTinputs"
         if not mySet == "":
            outFilePath = outFilePath + "_" + mySet
         if not suffix == "":
            outFilePath = outFilePath + "_" + suffix
         outFilePath = outFilePath + "_standardized.h5"
         outF = h5py.File(outFilePath, "w")
         print("Creating Standarized Dataset for ", mySample, len(jetBESDF[mySample]))
         outF.create_dataset('BES_vars', data=jetBESDF[mySample], compression='lzf')

         inFilePath = fileDir+mySample+"Sample_BESTinputs"
         if not mySet == "":
            inFilePath = inFilePath + "_" + mySet
         if not suffix == "":
            inFilePath = inFilePath + "_" + suffix
         inFilePath = inFilePath + ".h5"
         inF = h5py.File(inFilePath, "r")
         #Copy the images to the new file
         #Treat QCD separately because of dumb labeling scheme I introduced
         for myFrame in ['HiggsFrame_images','TopFrame_images','ZFrame_images','WFrame_images']:
            print("Copying", myFrame)
            outF.create_dataset(myFrame, data=inF[myFrame], compression='lzf')
         inF.close()
         outF.close()
         print("Done creating", outFilePath)
      print("Finished making datasets for", mySet)

# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) --all; 2) W,Z,b',
                        required=True)
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="/uscms/home/bonillaj/nobackup/h5samples/")
    parser.add_argument('-sf','--suffix',
                        dest='suffix',
                        default="")
    parser.add_argument('-st','--setType',
                        dest='setType',
                        help='<Required> Which (comma separated) sets to process. Examples: 1) --all; 2) train,validation,test',
                        required=True)
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if not args.setType == "all": setTypes = args.setType.split(',')

    # Make directories you need
    if not os.path.isdir(args.h5Dir): print(args.h5Dir, "does not exist")

    standardizeBESTVars(args.h5Dir, listOfSamples, setTypes, args.suffix)
        
    ## Plot total pT distributions
    
    print("Done")

