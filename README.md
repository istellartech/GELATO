# GELATO: Generic Launch Trajectory Optimizer
![reviewdog](https://github.com/istellartech/Trajectory_Optimization/actions/workflows/reviewdog.yaml/badge.svg)
![license](https://img.shields.io/github/license/istellartech/GELATO)

GELATO is an open source tool for launch trajectory optimization, written in Python.

GELATO solves trajectory optimization problems using the Legendre-Gauss-Radau pseudospectral method, which is stable and robust for highly complex nonlinear problems.


### Features

 - Maximization of payload mass by changing the angular rate profile and event times (such as cutoff time)
 - 3DoF calculation of dynamics with thrust, aerodynamics and gravity
 - Support for multi-stage launch vehicles
 - User customizable constraints

## Install

This program uses the libraries below:
 - NumPy
 - SciPy
 - pandas
 - Numba
 - pyoptsparse (see below before installation)

This program also requires IPOPT or SNOPT as a NLP solver. Install them separately before use.

You must build pyoptsparse from source in order to call external NLP solvers such as IPOPT or SNOPT from it. Please refer to the [pyoptsparse instructions](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/IPOPT.html).




## Usage

### Local

Prepare input files in the root directory of the repository.
 - setting files
 - an event file
 - a user constraint file (arbitary)

Run
```
python3 Trajectory_Optimization.py [setting file name]
```

See the example folder for an example set of input files.


### Batch operation

Prepare a folder that contains input files.

Run
```
./run_batch.sh [input folder]
```

and the optimization program will run for ALL settings files (\*.json) in the input folder.


### via AWS

Put the input folder on your S3 bucket.

Run
```
./run_batch.sh [S3 input directory full path]
```


## References

 - David, Benson. (2005). A Gauss Pseudospectral Transcription for Optimal Control.
 - Garrido, José. (2021). Development of Multiphase Radau Pseudospectral Method for Optimal Control Problems.
 - Di Campli Bayard de Volo, G. (2017). Vega Launchers' Trajectory Optimization Using a Pseudospectral Transcription.
 - Garg, Divya & Patterson, Michael & Hager, William & Rao, Anil & Benson, David & Huntington, Geoffrey. (2009). An overview of three pseudospectral methods for the numerical solution of optimal control problems. Advances in the Astronautical Sciences. 135. 
 
