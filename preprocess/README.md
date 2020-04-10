# Preprocessing for BEST

This ED Producer preprocesses CMS Monte Carlo samples. After preprocessing, these datasets 
can be used to train BEST. In the context of this software package, preprocessing means
reducing the size of the input data set by organizing TTrees by jet, then performing preselection
on those jets and matching to gen particles, and finally calculating and storing only the variables
of interest to BEST.

## Overview

The actual producer is located in the ``plugins/BESTProducer.cc`` and
the run instructions are located in ``test/run_*.py``.

# Instructions for Preprocessing

The preprocessing program can be run locally or through CRAB.

## Local Instructions

To run, use the cms environment to run a ``run_*.py`` file. For example: 

```bash
cmsenv
cd test/
cmsRun run_ZZ_test.py
```

Be sure to update any file locations in the ``run_*.py`` files!!

## CRAB Instructions

First, set up the CRAB environment and obtain a proxy

```bash
cd test/submit20XX
cmsenv
source /cvmfs/cms.cern.ch/crab3/crab.sh
voms-proxy-init --voms cms --valid 168:00
``` 

Now submit any of the CRAB files.

```bash
crab submit crab_*.py
```

The output file should be ``BESTInputs.root``. DAS datasets can be updated inside the ``crab_*.py`` files.

### Useful CRAB Commands

To test, get estimates, and then submit do a crab dry run

```bash
crab submit --dryrun submit.py
crab proceed
```

To resubmit failed jobs

```bash
crab resubmit crab_projecs/<project_directory>
