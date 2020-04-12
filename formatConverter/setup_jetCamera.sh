#!/bin/bash
#=========================================================================================
# setup_jetCamera.sh ---------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Brendan Regnery -------------------------------------------------------------
# This shell script sets up a python virtual environment for use with the jet camera -----
#-----------------------------------------------------------------------------------------

# Detect when we're not being sourced, print a hint and exit
# Based on https://stackoverflow.com/questions/2683279/how-to-detect-if-a-script-is-being-sourced#34642589
# When "return" fails (ie if not sourced), an error message is printed and
# caught by the if clause.
# In the normal mode of operation (ie if sourced), "return" is silent
if [[ ! -z "$(return 2>&1)" ]];
then
    echo >&2 "ERROR: You must use \"source $0\" /path/to/venvs to run this script."
    kill -INT $$
fi

# Make sure a venv directory argument was passed
if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please give the path for the desired virtual environment directory"
fi

# print some dependency information
echo "Welcome to the virtual environment setup script"
echo "Please note that this only works with BASH currently"

# check that user passed a path for virtual environment
if [ ! -e $1 ]
then
    echo "Please pass a path for the directory containing your virtual environments" 
    echo "par exemple: 'source setup_jetCamera.sh /this/is/your/path/venvDir'"
    echo "make sure the path you passed actually exists"
    exit -1
fi

# make sure that there are plots and images directories
if [ ! -e ./plots ]
then
    mkdir ./plots
fi

if [ ! -e ./h5samples ]
then
    mkdir ./h5samples
fi

# set venv path
export VENV_PATH=$1/jetCamEnv

# activate the virtual environment if it exists
if [ -e $VENV_PATH ]
then
    echo "Activating the virtual environment " $VENV_PATH
    source $VENV_PATH/bin/activate

# make a virtual environment if it does not exist
else
    echo "Creating the virtual environment " $VENV_PATH
    virtualenv -p python $VENV_PATH
    source $VENV_PATH/bin/activate
    # install dependencies in the virtual environment
    #pip install -U "pip<10" importlib
    pip install -U setuptools
    pip install -U numpy
    pip install -U pandas
    pip install -U awkward
    pip install -U uproot_methods
    pip install -U uproot
fi  

