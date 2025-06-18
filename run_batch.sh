#!/bin/bash

# Move to GELATO folder
GELATO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$GELATO_DIR"

if [ $# -lt 1 ]; then
    echo "input directory is not specified."
    exit 1
fi

echo $1

# Create temporary folder (using timestamp to make it unique)
TEMP_DIR="temp_$(date +%Y%m%d_%H%M%S)"
mkdir $TEMP_DIR

# Cleanup function to return to original directory
cleanup() {
    cd "$GELATO_DIR"
    if [ -d "$TEMP_DIR" ]; then
        rm -rf $TEMP_DIR
    fi
}

# Execute cleanup on error or script termination
trap cleanup EXIT

# Move to temporary folder
cd $TEMP_DIR

# Copy necessary files to temporary folder
cp "$GELATO_DIR/_user_constraints_empty.py" user_constraints.py

# Copy input data to temporary folder
if [[ $1 =~ ^s3:// ]]; then
    echo "Checking if S3 path exists: $1"
    # Check if S3 path exists
    if ! aws s3 ls "$1/" --cli-read-timeout 30 --cli-connect-timeout 30 > /dev/null 2>&1; then
        echo "Error: S3 path does not exist or is not accessible: $1"
        echo "Please check the path and your AWS permissions."
        # List available paths for reference
        PARENT_PATH=$(dirname "$1")
        echo "Available paths in $PARENT_PATH:"
        aws s3 ls "$PARENT_PATH/" --cli-read-timeout 30 --cli-connect-timeout 30 | grep "PRE" | head -10
        exit 1
    fi
    
    echo "Downloading from S3: $1"
    # Add timeout and verbose output for S3 operations
    timeout 300 aws s3 cp $1 . --recursive --exclude "output*" --cli-read-timeout 60 --cli-connect-timeout 60
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download from S3 or operation timed out"
        exit 1
    fi
    echo "S3 download completed successfully"
else
    # Handle absolute or relative paths appropriately
    if [[ "$1" = /* ]]; then
        # Absolute path
        cp -r "$1"/* .
    else
        # Relative path (relative to GELATO folder)
        cp -r "$GELATO_DIR/$1"/* .
    fi
fi

# Create output folder (remove existing one if present)
if [ -d "output" ]; then
    rm -rf output
fi
mkdir output

# Process each JSON file
for file in `find . -maxdepth 1 -name '*.json'`; do
    echo $file
    # Execute in temporary folder with PYTHONPATH set
    PYTHONPATH=".:$GELATO_DIR:$PYTHONPATH" python3 "$GELATO_DIR/Trajectory_Optimization.py" $file
done

# Copy results back to original location
if [[ $1 =~ ^s3:// ]]; then
    echo "Uploading results to S3: $1/output"
    timeout 300 aws s3 cp output $1/output --recursive --cli-read-timeout 60 --cli-connect-timeout 60
    if [ $? -ne 0 ]; then
        echo "Error: Failed to upload to S3 or operation timed out"
        exit 1
    fi
    echo "S3 upload completed successfully"
else
    # Handle absolute or relative paths appropriately
    if [[ "$1" = /* ]]; then
        # Absolute path
        cp -r output "$1"
    else
        # Relative path (relative to GELATO folder)
        cp -r output "$GELATO_DIR/$1"
    fi
fi

# Cleanup is automatically executed by trap
