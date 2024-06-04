cmake_minimum_required(VERSION 3.13)

# Project name and version
project(MyProject VERSION 1.0)

# Specify the C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

# Add the CMSIS paths
include_directories(CMSIS/NN/Include)
include_directories(CMSIS/DSP_CORE)
include_directories(CMSIS/DSP)

# Add the CMSIS-NN source files
file(GLOB CMSIS_NN_SOURCES
    "CMSIS/NN/Source/ActivationFunctions/*.c"
    "CMSIS/NN/Source/BasicMathFunctions/*.c"
    "CMSIS/NN/Source/ConcatenationFunctions/*.c"
    "CMSIS/NN/Source/ConvolutionFunctions/*.c"
    "CMSIS/NN/Source/FullyConnectedFunctions/*.c"
    "CMSIS/NN/Source/NNSupportFunctions/*.c"
    "CMSIS/NN/Source/PoolingFunctions/*.c"
    "CMSIS/NN/Source/ReshapeFunctions/*.c"
    "CMSIS/NN/Source/SVDFunctions/*.c"
    "CMSIS/NN/Source/SoftmaxFunctions/*.c"
)

# Add the executable for your project
add_executable(my_project my_project.c ${CMSIS_NN_SOURCES})

# Link against the CMSIS-DSP library (if necessary)
target_link_libraries(my_project m)  # Link the math library if needed

