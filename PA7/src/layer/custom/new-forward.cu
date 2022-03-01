#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <set>
#include <utility>


#define BLOCK_WIDTH 16
#define BLOCK_DEPTH 4

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */


    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int ysize = B*M*H_out*W_out; 
    int xsize = B*C*H*W;
    int ksize = M*C*K*K; 

    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z; 
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    int row = blockDim.x * bx + tx;
    int col = blockDim.y * by + ty;
    int depth = blockDim.z * bz + tz; // which output layer are we working on?

    if (row >= 0 && row < H_out && col >= 0 && col < W_out && depth >= 0 && depth < M) {
        // printf("%d %d %d \n", row, col, depth);
        for (int b = 0; b < B; b++)  { // iterating through image in batch 
            y4d(b, depth, row, col) = 0;
            for (int c = 0; c < C; c++) { // iterating through input layers
                for (int p = 0; p < K; p++) { // iterating through the filter
                    for (int q = 0; q < K; q++) {
                        y4d(b, depth, row, col) += (x4d(b, c, row + p, col + q) * k4d(depth, c, p, q));
                    }
                }
            }
        }
    } else {
    }

}


	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k,
    float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C,
    const int H, const int W, const int K)

{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    int hout = H-K+1;
    int wout = W-K+1;
    int ysize = B*M*hout*wout; 
    int xsize = B*C*H*W;
    int ksize = M*C*K*K; 

    cudaMalloc(device_y_ptr, sizeof(float) * ysize); 
    cudaMalloc(device_x_ptr, sizeof(float) * xsize); 
    cudaMalloc(device_k_ptr, sizeof(float) * ksize); 

    cudaMemcpy(*(device_y_ptr), host_y, sizeof(float) * ysize, cudaMemcpyHostToDevice); 
    cudaMemcpy(*(device_x_ptr), host_x, sizeof(float) * xsize, cudaMemcpyHostToDevice); 
    cudaMemcpy(*(device_k_ptr), host_k, sizeof(float) * ksize, cudaMemcpyHostToDevice); 
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k,
    const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    int hout = H-K+1;
    int wout = W-K+1;
    int ysize = B*M*hout*wout; 
    int xsize = B*C*H*W;
    int ksize = M*C*K*K; 

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_DEPTH);
    dim3 dimGrid(ceil((float)(hout) / BLOCK_WIDTH), ceil((float)(wout) / BLOCK_WIDTH), ceil((float)(M) / BLOCK_DEPTH));
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K); 
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1); 
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, 
    const int B, const int M, const int C, const int H, const int W, const int K)

{
    // Copy the output back to host

    // Free device memory
    int hout = H-K+1;
    int wout = W-K+1;
    int ysize = B*M*hout*wout; 
    int xsize = B*C*H*W;
    int ksize = M*C*K*K; 

    cudaMemcpy(host_y, device_y, ysize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_y); 
    cudaFree(device_x); 
    cudaFree(device_k); 

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1); 
    }
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
