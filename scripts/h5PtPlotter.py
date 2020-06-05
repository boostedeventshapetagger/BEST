# modules
import ROOT as root
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevents opening displays (fast), must use before pyplot
import matplotlib.pyplot as plt
import h5py


# Global variables
listOfSamples = ["b","Higgs","QCD","Top","W","Z"]
listOfFileTypes = ["","_train","_validation","_test"]

# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    """
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) --all; 2) W,Z,b',
                        required=True)
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="h5samples/")
    parser.add_argument('-ft','--fileTypes',
                        dest='fileTypes',
                        default="")
    parser.add_argument('-d','--debug',
                        action='store_true')
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if not args.fileTypes == "all": listOfFileTypes = args.samples.split(',')
    if args.debug:
        print("Samples to process: ", listOfSamples)
        print("File types to process: ", listOfFileTypes)

    # Make directories you need
    if not os.path.isdir(args.h5Dir): print(args.h5Dir, "does not exist")
    """

    ## First plot all pt for each collection
    ## So full samples, then train,validation,test, then train_flattened,validation_flattened,test_flattened
    for collection in [[".h5"],["_train.h5","_validation.h5","_test.h5"],["_train_flattened.h5","_validation_flattened.h5","_test_flattened.h5"]]:
        for suffix in collection:
            myPtArrays = []
            for mySample in listOfSamples:
                inputFile = h5py.File("/uscms/home/bonillaj/nobackup/h5samples/"+mySample+"Sample_BESTinputs"+suffix,"r")
                myPtArrays.append(np.array(inputFile["BES_vars"][...,35]))
            # --- Create histogram, legend and title ---
            plt.figure()
            H = plt.hist(myPtArrays, label=listOfSamples, normed=True)
            leg = plt.legend(frameon=False)
            plt.show()
            plt.savefig("PtDistribution"+suffix.split('.')[0]+'_Normalized.png')
            plt.clf()
            
    ## Plot total pT distributions
    
    print("Done")

