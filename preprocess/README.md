# Preprocessing for BEST

This ED Producer preprocesses CMS Monte Carlo samples. After preprocessing, these datasets 
can be used to train BEST. In the context of this software package, preprocessing means
reducing the size of the input data set by organizing TTrees by jet, then performing preselection
on those jets and matching to gen particles, and finally calculating and storing only the variables
of interest to BEST.

## Overview

The actual producer is located in the ``plugins/BESTProducer.cc`` and
the run scripts are located in ``test/submit201X/run_*.py``.

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
cd test/submit201X
cmsenv
source /cvmfs/cms.cern.ch/crab3/crab.sh
voms-proxy-init --voms cms --valid 168:00
``` 

Now submit any of the CRAB files.

```bash
crab submit crab_*.py
```

The output file should be ``BESTInputs.root`` in the eos location specified in the crab config script. DAS datasets can also be updated inside the ``crab_*.py`` files.

If you want to submit all crab files of a particular (or set of) sample, use the submitCrab.sh symbolic link within test/submit201X/submitCrab.sh (it link to the file BEST/scripts/submitCrab.sh). 
Include the samples, space separated, as positional arguments to the shell script. Use the --help function of the script for more details.

```bash
cd test/submit2017/
./submitCrab.sh HH tt QCD
```

If instead you would like to submit all crab jobs for a particular year use 'all' as the singular argument

```bash
cd test/submit2017/
./submitCrab.sh all
```

### Useful CRAB Commands

To test, get estimates, and then submit do a crab dry run

```bash
cd test/submit2017/
crab submit --dryrun crab_*.py
crab proceed
```

To check the jobs of a specific submission

```
cd test/submit2017/
crab status CrabBEST/<project_directory>
```

To check all jobs, use the shell script

```
cd test/submit2017/
./checkJobs.sh
```

To resubmit failed jobs

```bash
cd test/submit2017/
crab resubmit CrabBEST/<project_directory>
```

To kill a specific set of jobs of a submission

```
cd test/submit2017/
crab kill CrabBEST/<project_directory>
```

To kill all jobs, use the shell script

```
cd test/submit2017/
./killJobs.sh
```