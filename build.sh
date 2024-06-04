#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Navigate to the build directory
cd build

# Generate build files using CMake
cmake ..

# Build the project using make (assuming it's a Makefile-based project)
make

