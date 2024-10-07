# Minimum CMake version required
cmake_minimum_required(VERSION 3.12)

# Define the project
project(MSML)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Add pybind11 submodule (assuming it's in extern/pybind11)
add_subdirectory(extern/pybind11)

# Add include directories for headers
include_directories(${CMAKE_SOURCE_DIR}/CPP/linear_regression)

# Build the Python module using pybind11
pybind11_add_module(linear_regression
        CPP/linear_regression/linear_regression.cpp
)
