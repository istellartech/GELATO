# =============================================================================
# GELATO C++ / pybind11 Build System
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(GELATO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find dependencies
find_package(Eigen3 REQUIRED)

# Detect pybind11 CMake dir from the Python interpreter
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _pybind11_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_result
)
if(_pybind11_result EQUAL 0 AND EXISTS "${_pybind11_cmake_dir}")
  set(pybind11_DIR "${_pybind11_cmake_dir}")
endif()
find_package(pybind11 REQUIRED)

# Core static library (compiled once, shared by all modules)
add_library(gelato_core STATIC
  src/Air.cpp
  src/Earth.cpp
  src/gravity.cpp
  src/Coordinate.cpp
  src/iip.cpp
)
target_include_directories(gelato_core PUBLIC src)
target_link_libraries(gelato_core PUBLIC Eigen3::Eigen)
if(MSVC)
  target_compile_options(gelato_core PRIVATE /W4)
else()
  target_compile_options(gelato_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# Helper function to define each pybind11 module
function(add_gelato_module name source)
  pybind11_add_module(${name} src/${source})
  target_link_libraries(${name} PRIVATE gelato_core)
  if(MSVC)
    target_compile_options(${name} PRIVATE /W4)
  else()
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  set_target_properties(${name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/lib"
  )
endfunction()

# pybind11 extension modules
add_gelato_module(USStandardAtmosphere_c pybind_USStandardAtmosphere.cpp)
add_gelato_module(coordinate_c           pybind_coordinate.cpp)
add_gelato_module(dynamics_c             pybind_dynamics.cpp)
add_gelato_module(utils_c                pybind_utils.cpp)
add_gelato_module(IIP_c                  pybind_IIP.cpp)
