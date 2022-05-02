#!/bin/bash
source /opt/intel/oneapi/setvars.sh

cd ~/Trajectory_Optimization

if [ $# -lt 1 ]; then
    echo "input directory is not specified."
    exit 1
fi

echo $1
rm *.json
rm *.csv

cp _user_constraints_empty.py user_constraints.py
if [[ $1 =~ ^s3:// ]]; then
    rm -r output
    aws s3 cp $1 . --recursive --exclude "output*"
else
    cp $1/* .
fi

mkdir output
for file in `\find . -maxdepth 1 -name '*.json'`; do
    echo $file
    python3 Trajectory_Optimization.py $file
done

if [[ $1 =~ ^s3:// ]]; then
    aws s3 cp output $1/output --recursive
    rm -r output
else
    mv output $1
fi

rm ./*.json
rm ./*.csv
rm ./user_constraints.py
