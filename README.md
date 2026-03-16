# GELATO: Generic Launch Trajectory Optimizer
![reviewdog](https://github.com/istellartech/Trajectory_Optimization/actions/workflows/reviewdog.yaml/badge.svg)
![license](https://img.shields.io/github/license/istellartech/GELATO)

GELATO is an open source tool for launch trajectory optimization, written in Python.

GELATO solves trajectory optimization problems using the Legendre-Gauss-Radau pseudospectral method, which is stable and robust for highly complex nonlinear problems.


### Features

 - Maximization of payload mass by changing the angular rate profile and event times (such as cutoff time)
 - 3DoF calculation of dynamics with thrust, aerodynamics and gravity
 - Support for multi-stage launch vehicles
 - Automatic computation of Jacobians using the CasADi framework

## Install

This program uses the libraries below:
 - NumPy
 - SciPy
 - CasADi
 - Pandas
 - Simplekml (Optional: for tools/make_kml.py only)
 - Matplotlib (Optional: for tools/plot_output.py only)
 - Flask (Optional: for tools/settings_editor.py only)

The IPOPT solver for NLP is included with CasADi. If you wish to use SNOPT as the solver, please set the environment variables appropriately. For Linux, add the location of `libsnopt7_cpp.so` to `LD_LIBRARY_PATH`.

## Usage

Prepare input files in the root directory of the repository.
 - Setting JSON file
 - Related setting files (initial trajectory, wind, aerodynamics coefficients, etc.)

You can also create and edit input files using the GUI by running `tools/settings_editor.py`.

Run
```
make
python3 Trajectory_Optimization.py [setting file name]
```

See the example folder for an example set of input files.

## References

 - David, Benson. (2005). A Gauss Pseudospectral Transcription for Optimal Control.
 - Garrido, José. (2021). Development of Multiphase Radau Pseudospectral Method for Optimal Control Problems.
 - Di Campli Bayard de Volo, G. (2017). Vega Launchers' Trajectory Optimization Using a Pseudospectral Transcription.
 - Garg, Divya & Patterson, Michael & Hager, William & Rao, Anil & Benson, David & Huntington, Geoffrey. (2009). An overview of three pseudospectral methods for the numerical solution of optimal control problems. Advances in the Astronautical Sciences. 135. 
 