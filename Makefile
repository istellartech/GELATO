
CC = g++
CFLAGS = -O3 -Wall -shared -std=c++11 -I/usr/include/eigen3 -fPIC -rdynamic
PYBIND = `python3 -m pybind11 --includes`
PYCONFIG = `python3-config --extension-suffix`

SRCS = src/Air.cpp src/Earth.cpp src/gravity.cpp src/Coordinate.cpp

all: utils

utils: utils
	$(CC) $(CFLAGS) $(PYBIND) $(SRCS) src/pybind_USStandardAtmosphere.cpp -o USStandardAtmosphere_c$(PYCONFIG)
	$(CC) $(CFLAGS) $(PYBIND) $(SRCS) src/pybind_coordinate.cpp -o coordinate_c$(PYCONFIG)