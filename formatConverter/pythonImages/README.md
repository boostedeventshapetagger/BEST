# Jet Camera

## Overview

The Jet Camera is used to produce boosted jet images and store BES variables that are later used to train 
the BEST neural network

# Camera Instructions

Boosted jet images must first be created and then can be trained over. Make sure the file is correctly updated with the location of the preprocessed `.root` files.

## Image Creation

The images can be created with the appropriate .root files from the preprocessing step. To create the images, run
one of the ``imageCreator.py`` files in the CMS environment.

```bash
cmsenv
python imageCreator.py
```

