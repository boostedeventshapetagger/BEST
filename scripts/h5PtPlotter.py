# modules
import ROOT as root
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevents opening displays (fast), must use before pyplot
import matplotlib.pyplot as plt
import h5py
import argparse, os

# User definitons
bins_list = [i*100 for i in range(0,40)]



# Global variables
listOfSamples = ["b","Higgs","QCD","Top","W","Z"]
listOfFileTypes = [".h5","_train.h5","_validation.h5","_test.h5","_train_flattened.h5","_validation_flattened.h5","_test_flattened.h5"]

# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='Which (comma separated) samples to process. Examples: 1) --all; 2) W,Z,b',
                        default="all")
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        help='Location of directory containing h5 files to plot',
                        default="/uscms/home/bonillaj/nobackup/h5samples/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        help='Location of destination directory containing plots',
                        default="plots/")
    parser.add_argument('-ft','--fileTypes',
                        dest='fileTypes',
                        help='Which (comma separated) samples to process. Examples: 1) --all; 2) _train,_test',
                        default="all")
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if not args.fileTypes == "all": listOfFileTypes = args.fileTypes.split(',')
    outDir = args.outDir
    if not outDir[-1] == "/":
        outDir = outDir+"/"
    if not os.path.isdir(outDir):
      os.mkdir(outDir)
    print("Samples to process: ", listOfSamples)
    print("File types to process: ", listOfFileTypes)

    # Make directories you need
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()
    
    ## First plot all pt for each collection
    ## So full samples, then train,validation,test, then train_flattened,validation_flattened,test_flattened
    for suffix in listOfFileTypes:
        print("Plotting suffix", suffix)
        myPtArrays = []
        for mySample in listOfSamples:
            inputFile = h5py.File(args.h5Dir+mySample+"Sample_BESTinputs"+suffix,"r")
            myPtArrays.append(np.array(inputFile["BES_vars"][...,28]))
        # --- Create histogram, legend and title ---
        plt.figure()
        if suffix == ".h5":
            H = plt.hist(myPtArrays, bins = bins_list, histtype='step', log=True, label=listOfSamples, stacked=False, fill=False, normed=False)
            plt.ylim(top=10000000000)  # adjust the top leaving bottom unchanged
            plt.ylim(bottom=0.1)  # adjust the bottom leaving top unchanged
        else:
            H = plt.hist(myPtArrays, histtype='step', stacked=False, fill=False, bins = bins_list, label=listOfSamples, normed=False)
        leg = plt.legend(frameon=False)
        plt.show()
        plt.savefig(outDir+"PtDistribution"+suffix.split('.')[0]+'.png')
        plt.clf()
        # --- Normalized Create histogram, legend and title ---
        plt.figure()
        H = plt.hist(myPtArrays, histtype='step', stacked=False, fill=False, bins = bins_list, label=listOfSamples, normed=True)
        leg = plt.legend(frameon=False)
        plt.show()
        plt.savefig(outDir+"PtDistribution"+suffix.split('.')[0]+'_Normalized.png')
        plt.clf()

    
    print("Done")

