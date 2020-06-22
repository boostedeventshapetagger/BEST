#!/bin/bash
#=========================================================================================
# listQCDsamples.sh ----------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Johan S Bonilla, Brendan Regnery --------------------------------------------
#-----------------------------------------------------------------------------------------

# List files from eos with BEST in name (typically /eos/path/BESTInputs_*.root)
eosDirPath="/store/user/jbonilla/"
echo "Listing files in $eosDirPath"
eosBESTFiles=`xrdfsls -R $eosDirPath | grep '.*QCD.*BEST'`

# Check if file exists, if so delete
fileToWrite="./listOfQCDfilePaths.txt"
if [ -f $fileToWrite ] ; then
    rm $fileToWrite
fi

# Write each BESTInput-file's xrootd path to fileToWrite
for f in $eosBESTFiles; do
    echo "root://cmsxrootd.fnal.gov/$f" >> $fileToWrite
done

echo "Checkout your new list of files at $fileToWrite"

