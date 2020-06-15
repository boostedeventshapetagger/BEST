# Training the Boosted Event Shape Tagger

The programs in this directory train the BEST.

## Overview
There are a few different training strategies that live in this directory.

``trainHHESTIA.py`` can produce plots of the input variables and training results. These features can be turned on and off with the boolean variables at the beginning of the program.

``trainWithGenerators.py`` uses batch-generation for feeding in data.

``johanTraining.py`` uses the shape-matching technique.

## Using the FermiLab GPUs

BEST is a complicated network that takes a lot of processing to train, so we recommend training it using a GPU. 
To use the GPU at FermiLab, set up an appropriate GPU environment. 
Do NOT use the CMS environment.

```bash
/bin/bash --login
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh
source activate mlenv0
```

## Training with Batch Generator pT flattening

Our current method of training BEST uses a batch generator for pT flattening. This training is done in three steps on the LPC GPU.

```bash
python MakeFlatWeights.py
python MakeStandardInputs.py -s all -hd </path/to/hdSamples/>
python TrainWithGenerators.py
```

## Training with Shape-Matching

The pT of the samples have been decorrelated (shape-matched) in the formatConverter step. Before the training, one must standardize the inputs for all file sets. Then one trains the network. Once done, one can evaluate the model on the test data. Thus the training and evaluation is done in three steps on the LPC GPU.

```bash
python MakeStandardInputs.py -s all -sf flattened -st train,validation,test
python TrainWithGenerators.py
python plotConfusionMatrix.py
```

## Training with only the Higgs Frame Images

A smaller network can be trained on the images. Various network options are available in the files titled ``imageTraining.py``.
Note this is for testing and has no pT flattening.

```bash
python imageTraining.py
```

## Warning About Functions in Python

Python does not forget about operations done to a variable inside a function. If a variable ``var`` is declared
in the main program and a function then deletes ``var`` in order to return something else, ``var`` will also be
deleted from the main program. This also includes any variables that point to the same memory; for example 
``var2 = var`` will also be deleted. To avoid this, use the copy module to copy the memory.

```python
import copy
var2 = copy.copy(var)
result = function(var) # function that deletes var
# var2 will still be here, but var will be deleted
```

