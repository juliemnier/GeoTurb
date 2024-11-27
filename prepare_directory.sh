#!/bin/bash

#>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Path to run:
WHERE_TO=./run_test
#...........................
#>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Directory of GeoTurb.jl:
#...........................
GT_ROOT=./GeoTurb/

if [ -d "$WHERE_TO" ]; then

#Check whether the directory exists
#in which case, cancel the creation

echo " ..."
echo " >>> !!! This directory already exists !!! <<< "
echo " ..."
echo " Aborting the creation of directory/subdirectories"
echo " Delete the directory manually before you can create"
echo " it with this script."

else

# If the directory does not exist,
# proceed with creation

echo " ..."
echo " Creating the directory:"
echo ${WHERE_TO}

mkdir ${WHERE_TO}
mkdir ${WHERE_TO}/Snapshots
mkdir ${WHERE_TO}/Fields
cp ${GT_ROOT}/utils/main.jl ${WHERE_TO}
cp ${GT_ROOT}/utils/namelist.jl ${WHERE_TO}
cd ${WHERE_TO}
fi
