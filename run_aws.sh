#!/bin/bash
cd ~/Trajectory_Optimization

if [ $# -lt 1 ]; then
    echo "input directory is not specified."
    exit 1
fi

echo $1
aws s3 cp s3://ist-system/Trajectory_Optimization/$1 . --recursive --exclude "output*"

ls *.json | xargs python3 Trajectory_Optimization.py

aws s3 cp output s3://ist-system/Trajectory_Optimization/$1/output --recursive

rm -r output
rm ./*.json
rm ./*.csv
cp _user_constraints_empty.py user_constraints.py