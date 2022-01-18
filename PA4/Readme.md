
# Programming Assignment 4: CUDA Tiled Matrix Multiplication

## Objective

Implement a tiled dense matrix multiplication routine using shared memory.

## Instructions

Edit the code in the code tab to perform the following:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- copy results from device to host
- deallocate device memory
- implement the matrix-matrix multiplication routine using shared memory and tiling

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

## How to Compile

The `template.cu` file contains the code for the programming assignment. There is a Makefile included which compiles it and links it with the libgputk CUDA library automatically. It can be run by typing `make` from the VectorAdd folder. It generates a `solution` output file. During development, make sure to run the `make clean` command before running `make`. 

## How to test

Use the `make run` commad to test your program. There are a total of 9 tests on which your program will be evaluated.

## Dataset Generation (optional)

The dataset required to test the program is already generated. If you are interested in how the dataset is generated please refer to the `dataset_generator.cpp` file. You may compile this file using the `make dataset_generator` command and run the executeable using the command `./dataset_generator`. 

## Submission

Submit the template.cu file on gradescope.