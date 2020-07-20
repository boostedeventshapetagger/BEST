# Format Converter

## Overview

The Format Converter creates python h5 files from input root ntuples so that the images and BES variables are in a proper format for training the BEST neural network.

The `pythonImages` directory contains files for creating 
jet images using python functions; this information is kept for developers who want to test new image techniques.

The `tools` directory contains functions for plotting jet images and making python comparison images.

The `test_boost_jetImageCreator.py` file is used by the CI to test that the images are being properly created. 
It compares the C++ images to ones created in python.

There are two strategies for training (shape-matching and batch-generator), which require slightly different steps to produce the data needed to train.

## Tagged Samples

If you wish to use already produced samples, copy them from eos

```
mkdir ~/nobackup/BESTTag2Samples
eosls /store/user/jbonilla/BESTTag2Samples/
xrdcp root://cmsxrootd.fnal.gov//store/user/jbonilla/BESTTag2Samples/<FileYouWishToCopy> ~/nobackup/BESTTag2Samples/
```

## Conversion Instructions

The conversion takes place using uproot to create useful python data structures. First, make sure that there are directories to store
the h5 files and image plots. Consider making the output h5sample directory in your nobackup space.

```bash
mkdir plots
mkdir ~/nobackup/h5samples
```

Make sure that you have `cmsenv` enabled and have a `vprox`. Next, the lists of eos files must be created using the
`eosSamples/listXSamples.sh`. If you have created new `.root` ntuples, be sure to update the search paths in these
shell scripts.

```bash
cd eosSamples
source listXSamples.sh
```

For batch generation, `X_sample_formatConverter.py`files. 

```bash
python Higgs_sample_formatConverter.py
<For each sample>
```

For shape-matching, use the more generic 'sampleConverter.py' script. This may also work for batch-generation but not tested.

```bash
python sampleConverter.py -s all -eos <path/to/eosSampleListsDir/> -o <path/to/h5samples/>
```

Also for shape-matching, you will want to first split the sample into train, validation, and test sets with 'sampleSplitter.py' then flatten (shape-match) each of those output files. See each file for full list of arguments.

```bash
python sampleSplitter.py -s all -hd </path/to/h5samples/> -o </path/to/outputH5samples> -bs <batchSize=(example)600000>
python flattener.py -s all -st train,validation,test -hd </path/to/Inputh5samples/> -o </path/to/Outputh5samples/> -b <batchSize=(example)250000>
```


## (Optional) Python Virtual Environment

A set up script for a python virtual environment is included for any developers who need access to the most up-top-date
python tools. To use it, do the following.

```bash
# make sure that you are using bash (not tcsh or zsh)
cmsenv
source ./setup_jetCamera.sh /path/to/venvs
```

To exit the virtual environment.

```bash
deactivate
```
