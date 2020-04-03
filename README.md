# BEST: Boosted Event Shape Tagger

## Dependencies 

This repository requires CMSSW and python tools for machine learning.

## Installation

This program is written for use with ``CMSSW_10_2_18``. Start installation by installing CMSSW.

```bash
cmsrel CMSSW_10_2_18
cd CMSSW_10_2_18/src/
scram b -j8
```
Then, make a fork, clone the repository, and compile the programs as modules for CMSSW.

```bash
cd CMSSW_9_4_8/src/
git clone https://gitlab.cern.ch/username/BEST.git
scram b -j8
```

Now the repository can be used. 

## Overview

Before training the neural network, the CMS datasets must be converted into a usable form.
To do this, see the instructions in the ``preprocess`` directory.
After preprocessing, the images need to be produced with the ``jetCamera``. Finally,
the files can be used to train a neural network. For training,
see the instructions in the ``training`` directory.

## Instructions for Contributing to this Repository

First, fork this repository and push code to the forked version.
Please only submit pull requests to the `developer` branch. Before submitting a pull request, 
please test your code. To test any changes to the preprocess step or jetCamera, please do the following:

```bash
cd BEST/preprocess/
cmsenv
scram b -j8
cmsRun test/run_ZZ_test.py
python BES_variable_testingSuite.py 
cd ../jetCamera
python test_boost_jetImageCreator.py
```

Then open up the output root file and make sure that the results are as expected. There are no
tests yet for any of the training files. So please keep old, stable training code in the `legacy` 
folder and create a new file when you make changes. 

After tests, please rebase to the current developer version:

```bash
# if this is your first time submitting a pull request, then do
git remote add BEST https://gitlab.cern.ch/boostedeventshapetagger/BEST.git
git fetch -p --all
git checkout -b CentralDev -t BEST/developer #this creates a local branch called CentralDev that tracks the main developer branch
# then every time you want to ensure that the code is up to date
git fetch -p --all
git checkout CentralDev
git pull
git checkout feature/MyFeatureBranch
git rebase -i CentralDev
# follow the rebase instructions
git push 
```

Finally, submit your a merge request on GitLab to the `developer` branch in `boostedeventshapetagger/BEST`.
There is a short form to fill out for the pull request, this will help the maintainers understand your changes.
Then, your changes will be reviewed before being added. 

#### To make a new feature branch

Make sure that you are up-to-date with BEST/developer before making a new feature branch

```bash
git fetch -p --all
git checkout CentralDev
git pull
git checkout -b feature/MyFeatureBranch
```

## NTuple location

Some preprocessed ntuples of the Monte Carlo simulated data already exist on the LPC EOS at `/store/user/rband/BESTSamples` These ntuples can be used in the JetCamera step.

