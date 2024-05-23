
CC = g++
CFLAGS = -O3 -Wall -shared -std=c++11 -I/usr/include/eigen3 -fPIC
PYBIND = `python3 -m pybind11 --includes`
PYCONFIG = `python3-config --extension-suffix`

all: utils

utils: utils
	$(CC) $(CFLAGS) $(PYBIND) src/Air.cpp src/USStandardAtmosphere.cpp -o USStandardAtmosphere_c$(PYCONFIG)
