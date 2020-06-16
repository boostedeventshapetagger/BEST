#!/bin/bash
#=========================================================================================
# submitCrab.sh --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Johan S Bonilla -------------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script takes arguments (sample keys) from the user and submits the appropriate jobs to crab.
# The script lives in the scripts directory and the symbolic links in each of the submit201X directories should be executed within their respective directories.

# Check that user provided arguments, exit if not.
if [ $# -gt 0 ]; then
    echo "Your command line contains $# arguments."
else
    echo "Your command line contains no arguments, please specify what keys you'd like to submit."
    echo "Options: QCD, HH, WW, ZZ, tt, bb, all"
    echo "Example: ./submitCrab.sh HH WW tt"
    echo "Example: ./submitCrab.sh all"
    exit 1
fi

# Extract keys from argument(s)
declare -a myKeys
if [ $1 == "all" ]; then
    echo "Submitting all samples"
    myKeys=("HH" "WW" "ZZ" "tt" "bb" "QCD")
else
    for arg; do
	myKeys+=($arg)
    done
fi
echo "Submitting samples: ${myKeys[@]}"

# Check if logFiles directory exists, else make one
if [[ -d "logFiles" ]]
then
    echo "logFiles exists on your filesystem."
else
    mkdir logFiles
    echo "Created logFiles directory."
fi

# Find all crab scripts of each key and submit them in parallel
for myKey in ${myKeys[@]}; do
    echo "Finding submition scripts for $myKey"
    listOfScripts=crab*$myKey*.py
    for f in $listOfScripts; do
	crabName=$(echo $f | cut -d'.' -f 1)
	crab submit $f >> logFiles/$crabName.txt &
    done
done
