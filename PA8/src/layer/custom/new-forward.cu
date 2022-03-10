#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <set>
#include <utility>
#include <unordered_set>


#define BLOCK_WIDTH 16
#define KERNEL_LENGTH 7


// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


__device__ bool in_bounds(int row, int col, int rBound, int cBound) {
    if (row >= 0 && row < rBound && col >= 0 && col < cBound) {
        return true; 
    }
    return false; 
}
__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    __shared__ float shared_x[BLOCK_WIDTH+KERNEL_LENGTH-1][BLOCK_WIDTH+KERNEL_LENGTH-1][4]; 

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int h_grid = ceil((float)(H_out) / BLOCK_WIDTH); 
    int w_grid = ceil((float)(W_out) / BLOCK_WIDTH); 

    int tx = threadIdx.x, ty = threadIdx.y; 
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;


    int batch_image = bz; 
    int m = bx; 

    int h = (by / w_grid) * BLOCK_WIDTH + tx; 
    int w = (by % w_grid) * BLOCK_WIDTH + ty; 

    if (in_bounds(h, w, H_out, W_out)) {
        for (int i = 0; i < C; i++) {
            shared_x[tx][ty][i] = x4d(batch_image, i, h, w); 
        }
        // if bottom
        bool lmao0 = (h == H_out-1); 
        bool lmao1 = (w == W_out-1); 
        if (tx == BLOCK_WIDTH-1 || h == H_out-1) {
            for (int i = 1; i <= K-1; i++) {
                for (int j = 0; j < C; j++) {
                    shared_x[tx+i][ty][j] = x4d(batch_image, j, h+i, w); 
                }
            }
        }
        // if all the way to the right
        if (ty == BLOCK_WIDTH-1 || w == W_out-1) {
            for (int i = 1; i <= K-1; i++) {
                for (int j = 0; j < C; j++) {
                    shared_x[tx][ty+i][j] = x4d(batch_image, j, h, w+i); 
                }
            }
        }

        // that means is a corner
        if ((tx == BLOCK_WIDTH-1 && ty == BLOCK_WIDTH-1) || (h == H_out-1 && w == W_out-1)) {
            for (int i = 1; i <= K-1; i++) {
                for (int j = 1; j <= K-1; j++) {
                    for (int c = 0; c < C; c++) {
                        shared_x[tx+i][ty+j][c] = x4d(batch_image, c, h+i, w+j); 
                    }
                }
            }
        }
        // (tx == BLOCK_WIDTH-1 || h == H_out-1) && (ty == BLOCK_WIDTH-1 || w == W_out-1)
        // ((tx == BLOCK_WIDTH-1 && ty == BLOCK_WIDTH-1) || (h == H_out-1 && w == W_out-1)) 
    }
    __syncthreads();
    if (in_bounds(h, w, H_out, W_out)) {
        y4d(batch_image, m, h, w) = 0; 
        for (int c = 0; c < C; c++) { 
            for (int p = 0; p < K; p++) {  
                for (int q = 0; q < K; q++) {
                    int shared_row = tx + p; 
                    int shared_col = ty + q; 

                    float shared = shared_x[shared_row][shared_col][c]; 
                    // float regular = x4d(batch_image, c, h +p, w+q); 
                    // all of them has either a 33 or 32 in it in the 40 case
                    /*

                        if (shared != regular) {
                            printf("%d Shared: %f Regular: %f Tilerow: %d Tilecol: %d Actualrow: %d Actualcol: %d Inputimage map: %d sharedrow: %d sharedcol: %d\n", H, shared_x[shared_row][shared_col][c], x4d(batch_image, c, h +p, w+q), tx, ty, h, w, c, shared_row, shared_col); 
                        }
                    */ 
                    y4d(batch_image, m, h, w) += shared * k4d(m, c, p, q); 

                    // normal version without shared memory
                    // y4d(batch_image, m, h, w) += x4d(batch_image, c, h +p, w+q) * k4d(m, c, p, q); 
                }
            }
        }
        // printf("%f\n", y4d(batch_image, m, h, w)); 
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
    printf("Number of images: %d\n", B); 
    printf("Number of output feature maps: %d\n", M); 
    printf("Number of input feature maps: %d\n", C); 
    printf("Height: %d\n", H); 
    printf("Width: %d\n", W); 
    printf("Size of filter: %d\n", K); 
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

    int h_grid = ceil((float)(hout) / BLOCK_WIDTH); 
    int w_grid = ceil((float)(wout) / BLOCK_WIDTH); 
    int num_tiles = h_grid * w_grid; 

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid(M, num_tiles, B); 
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
