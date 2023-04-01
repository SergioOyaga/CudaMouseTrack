# CudaMouseTrack
A basic example and demonstration on what a JCuda program looks like. :octocat: :octocat:

This project is an example of CUDA programming using Java and de NVIDIA Runtime Compiler (NVRTC).

<table>
  <tr>
    <td> <b>Mouse Track Example using JCuda </b></td>
    <td> <b>Mouse Track Example using JCuda with Spline interpolation</b> </td>
  </tr>
  <tr>
    <td> <img src="https://github.com/SergioOyaga/CudaMouseTrack/blob/master/out/image/MouseTracker-2022-07-10-14-21-17.gif"  title="Mouse Track Example" alt="Mouse Track Example" width="400" height="400" /></td>
    <td><img src="https://github.com/SergioOyaga/CudaMouseTrack/blob/master/out/image/MouseTracker-2022-07-25-18-07-30.gif"  title="Mouse Track Example With Spline" alt="Mouse Track Example With Spline" width="400" height="400" /></td>
  </tr>
</table>

## Table of Contents

* [Introduction](#introduction)
* [Java classes](#java-classes)
    * [JcudaMouseTracker](#jCudaMouseTracker)
    * [JcudaMouseTrackerSpline](#jCudaMouseTrackerSpline)
    * [ColorKernel](#colorKernel)
    * [VectorKernel](#vectorKernel)
    * [Spline](#spline)
* [CUDA kernels](#cuda-kernels)
    * [JCudaColorChanger](#JCudaColorChanger)
    * [JCudaVectorValueEdition](#JCudaVectorValueEdition)

## Introduction

This project is an introduction to CUDA in Java using the JCuda package which implements most of the typical cuda functions and capabilities.
It allows the user to implement from scratch the basis of cuda. In addition, some higher level features are implemented in the [_JCuda project_](http://javagl.de/jcuda.org/).
To see some basic examples and JCuda implementations, refer to the [_JCuda-samples github_](https://github.com/jcuda/jcuda-samples).

JCuda higher level packages/bindings for CUDA libraries:

- [**JCurand**](https://github.com/jcuda/jcurand)

  > Java bindings for cuRAND, the NVIDIA CUDA random (RAND) number generation library.

- [**JCublas**](https://github.com/jcuda/jcublas)
  >  Java bindings for cuBLAS, the NVIDIA CUDA basic linear algebra subroutines (BLAS) library.

- [**JCuff**](https://github.com/jcuda/jcufft)
  > Java bindings for cuFFT, the NVIDIA CUDA fast Fourier Transform (FFT) library.

- [**JCusparse**](https://github.com/jcuda/jcusparse)
  > Java bindings for cuSPARSE, the NVIDIA CUDA sparse matrix (SPARSE) library.

- [**JCusolver**](https://github.com/jcuda/jcusolver)
  > Java bindings for cuSOLVER, the NVIDIA CUDA GPU accelerated library for decompositions and linear system solutions for both dense and sparse matrices (SOLVER).

- [**JCudnn**](https://github.com/jcuda/jcudnn)
  > Java bindings for cuDNN, the NVIDIA CUDA Deep Neural Network (DNN) library.
  
- [**JCuda-Samples**](https://github.com/jcuda/jcuda-samples)
  > Java examples of JCuda and the rest of bindings.
  
Web documentation:

- [**NVIDIA CUDA @ web**](https://developer.nvidia.com): CUDA website.
- [**NVIDIA CUDA Libraries @ web**](https://developer.nvidia.com/gpu-accelerated-libraries): CUDA libraries website.


## Java classes
Here we can see all the java classes implemented in this example. 

Notice that we tried to left behind the C code style and implement a more Java friendly code. 
For the sake of simplicity, although some code could be implemented in abstract classes, 
with private methods and all the OOP stuff, we kept it as simple as possible even with redundant code.

### JcudaMouseTracker
This is the main class. Here we define the problem parameters, initialize constants and vectors, and create the kernels 
objects.

Here is also implemented the infinite loop where a thread is waiting for be awakened. This loop runs the kernel, and it 
is triggered by a mouse motion event.
A closing event is also captured to free GPU reserved memory.

### JcudaMouseTrackerSpline
This is the main class. Here we define the problem parameters, initialize constants and vectors, and create the kernels
objects.

Here is also implemented the infinite loop where a thread is waiting for be awakened. This loop runs the kernel, and it
is triggered by a mouse motion event. Onece the even is alerted, the interpolation of points take place.
A closing event is also captured to free GPU reserved memory.

### ColorKernel
It is a container class. The cuda characteristics such as grid/block sizes and cuda's function is stored to be launched 
by a  runKernel method.

In addition, this class (not like the [VectorKernel](#vectorkernel)) is in charge of set the cuda exceptions and the cuda
context (for the GPU) where the module programs will run.

This class objective is to compute the color of each pixel of our image (AKA vector).

### VectorKernel
It is a container class. The cuda characteristics such as grid/block sizes and cuda's function is stored to be launched
by a  runKernel method.

This class objective is to change in place the value of vector position directly in the GPU/CPU shared memory.

### Spline
This class contains the methods to create a cubic spline interpolator. It allows the user to interpolate directly using 
a static method, or an object of the class can be instantiated to call a method to interpolate new values.

This class objective is to change in place the value of vector position directly in the GPU/CPU shared memory.

## CUDA kernels
Here we can see all the CUDA kernels implemented in this example.

Notice that here the native CUDA C language is used.
For the sake of simplicity, although the code could be implemented in one unique .cu file, we preferred to use one 
function per file.

### JCudaColorChanger
This code contains a kernel definition that computes the color of an array according to the distance to the N last mouse
positions.
### JCudaVectorValueEdition
This code contains a kernel definition that replaces the value of a specific vector position by a specific value.
## Software
This code has been developed with:
- NVIDIA GeForce RTX 3070.
- openjdk-18.
- java 18.0.1.1.

## Disclaimer
Feel free to download, use or edit this code under your own responsibility.