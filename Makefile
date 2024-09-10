
CC = g++
CFLAGS = -O2 -Wall -shared -std=c++11 -I/usr/include/eigen3 -fPIC -rdynamic
PYBIND = `python3 -m pybind11 --includes`
PYCONFIG = `python3-config --extension-suffix`

SRCS = src/Air.cpp src/Earth.cpp src/gravity.cpp src/Coordinate.cpp src/iip.cpp

all: atmosphere coordinate dynamics utils iip

atmosphere:
	$(CC) $(CFLAGS) $(PYBIND) $(SRCS) src/pybind_USStandardAtmosphere.cpp -o lib/USStandardAtmosphere_c$(PYCONFIG)

coordinate:
	$(CC) $(CFLAGS) $(PYBIND) $(SRCS) src/pybind_coordinate.cpp -o lib/coordinate_c$(PYCONFIG)

dynamics:
	$(CC) $(CFLAGS) $(PYBIND) $(SRCS) src/pybind_dynamics.cpp -o lib/dynamics_c$(PYCONFIG)

utils:
	$(CC) $(CFLAGS) $(PYBIND) $(SRCS) src/pybind_utils.cpp -o lib/utils_c$(PYCONFIG)

iip:
	$(CC) $(CFLAGS) $(PYBIND) $(SRCS) src/pybind_IIP.cpp -o lib/IIP_c$(PYCONFIG)
