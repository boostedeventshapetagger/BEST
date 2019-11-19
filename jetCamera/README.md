# Jet Camera

## Overview

The Jet Camera is used to produce boosted jet images and store BES variables that are later used to train 
the BEST neural network

## Dependencies

The jet camera depends on uproot, which is only included with `CMSSW_10_X` and later. If you are using `CMSSW_9_4_X` MC samples,
then complete the preprocessing step and rebuild BEST in `CMSSW_10_X` to use the jet camera. 

```bash
cmsrel CMSSW_10_X
cd CMSSW_10_X/src/
git clone https://username@gitlab.cern.ch/boostedeventshapetagger/BEST.git
```

# Camera Instructions

Boosted jet images must first be created and then can be trained over. Make sure the file is correctly updated with the location of the preprocessed `.root` files.

## Image Creation

The images can be created with the appropriate .root files from the preprocessing step. To create the images, run
one of the ``imageCreator.py`` files in the CMS environment.

```bash
cmsenv
python imageCreator.py
```

