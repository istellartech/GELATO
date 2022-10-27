# Trajectory_Optimization
Rocket trajectory optimization

![reviewdog](https://github.com/istellartech/Trajectory_Optimization/actions/workflows/reviewdog.yaml/badge.svg)
## Install

This program uses the libraries below:
 - NumPy
 - SciPy
 - pandas
 - Numba
 - pyoptsparse(see below before installation)

This program also requires IPOPT or SNOPT as NLP solver. Install them separately before use.

You must build pyoptsparse from source in order to call external NLP solver such as IPOPT or SNOPT from it. Please refer to [pyoptsparse instruction](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/IPOPT.html).

## Usage

### Local

Prepare input files in the root directory of the repository.
 - setting file
 - event file
 - user constraint file(arbitary)

Run
```
python3 Trajectory_Optimization.py [setting file name]
```

### via AWS

Put input files in AWS S3 directory.

Execute
```
./run_batch.sh [S3 input directory full path]
```

and the optimization program will run for ALL settings files(\*.json) in the S3 directory

