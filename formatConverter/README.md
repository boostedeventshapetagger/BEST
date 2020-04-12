# Format Converter

## Overview

The Format Converter creates python h5 files from input root ntuples so that the images and BES variables are
in a proper format for training the BEST neural network. The `pythonImages` directory contains files for creating 
jet images using python functions; this information is kept for developers who want to test new image techniques.
The `tools` directory contains functions for plotting jet images and making python comparison images.

## Conversion Instructions

The conversion takes place using uproot to create useful python data structures. Make sure the file is correctly 
updated with the location of the preprocessed `.root` files. First, make sure that there are directories to store
the h5 files and image plots.

```bash
mkdir plots
mkdir h5samples
```

To run the conversion, use the `formatConverter.py`files. 

```bash
cmsenv
python formatConverter.py
```

The `test_boost_jetImageCreator.py` file is used by the CI to test that the images are being properly created. 
It compares the C++ images to ones created in python.

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
