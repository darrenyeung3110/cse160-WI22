#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

#define BLOCK_WIDTH 16

__device__ bool inBounds(int x, int y, int z, int xb, int yb, int zb) {
    if (x >= 0 && y >= 0 && z >= 0) {
        if (x < xb && y < yb && z < zb) {
            return true; 
        }
    } 
    return false; 
}
__global__ void  convolution(float *din, float *dmask, float *dout, int depth, int width, int height) {
    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z; 
    int bx = blockIdx.x, by = blockIdx.y; 

    /*
        These three variables stores the index of the image that we are computing
    */
    int row = blockDim.x * bx + tx; 
    int col = blockDim.y * by + ty; 
    int channel = tz;
    // printf("%d\n", channel); 
    if (inBounds(row, col, channel, height, width, depth)) {
        float accum = 0; 
        int collapsedIndex = (row * width + col) * depth + channel; 
        for (int i = -Mask_radius; i <= Mask_radius; i++) {
            for (int j = -Mask_radius; j <= -Mask_radius; j++) {
                int rowOffset = row + i; 
                int colOffset = col + j; 
                // check whether the offset is in bounds
                if (inBounds(rowOffset, colOffset, channel, height, width, depth)) {
                    int collapsedOffsetIndex = (rowOffset * width + colOffset) * depth + channel; 
                    int collapsedMaskIndex = (i + Mask_radius) * Mask_width + j + Mask_radius; 
                    float pixelChannelValue = din[collapsedOffsetIndex]; 
                    float maskValue = dmask[collapsedMaskIndex]; 
                    accum += pixelChannelValue * maskValue; 
                }
            }
        }
        dout[collapsedIndex] = clamp(accum); 
    }
}
//@@ INSERT CODE HERE
int main(int argc, char *argv[]) {
    gpuTKArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char *inputImageFile;
    char *inputMaskFile;
    gpuTKImage_t inputImage;
    gpuTKImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *hostMaskData;
    float *deviceInputImageData;
    float *deviceOutputImageData;
    float *deviceMaskData;

    arg = gpuTKArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = gpuTKArg_getInputFile(arg, 0);
    inputMaskFile  = gpuTKArg_getInputFile(arg, 1);

    inputImage   = gpuTKImport(inputImageFile);
    hostMaskData = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth    = gpuTKImage_getWidth(inputImage);
    imageHeight   = gpuTKImage_getHeight(inputImage);
    imageChannels = gpuTKImage_getChannels(inputImage);

    int collapsedSize = imageWidth * imageHeight * imageChannels; 
    int collapsedMaskSize = maskRows * maskColumns; 

    outputImage = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData  = gpuTKImage_getData(inputImage);
    hostOutputImageData = gpuTKImage_getData(outputImage);

    gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

    gpuTKTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void**)&deviceInputImageData, collapsedSize * (sizeof(float)));
    cudaMalloc((void**)&deviceOutputImageData, collapsedSize * (sizeof(float)));
    cudaMalloc((void**)&deviceMaskData, collapsedMaskSize * (sizeof(float)));
    //@@ INSERT CODE HERE
    gpuTKTime_stop(GPU, "Doing GPU memory allocation");

    gpuTKTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData, collapsedSize*(sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutputImageData, hostOutputImageData, collapsedSize*(sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData, collapsedMaskSize*(sizeof(float)), cudaMemcpyHostToDevice);
    //@@ INSERT CODE HERE
    gpuTKTime_stop(Copy, "Copying data to the GPU");

    gpuTKTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    int gridRows = ceil(((float)imageHeight) / BLOCK_WIDTH);
    int gridColumns = ceil(((float)imageWidth) / BLOCK_WIDTH);

    dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH, imageChannels); 
    dim3 gridSize(gridRows, gridColumns); 

    convolution<<<gridSize, blockSize>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
    imageChannels, imageWidth, imageHeight);

    gpuTKTime_stop(Compute, "Doing the computation on the GPU");

    gpuTKTime_start(Copy, "Copying data from the GPU");
    //@@ INSERT CODE HERE
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
    imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
    /*
        for (int i = 0; i < imageHeight; i++) {
            for (int j = 0; j < imageWidth; j++) {
                for (int z = 0; z < imageChannels; z++) {
                    int idx = i * imageWidth + j + z;
                    printf("%f ", hostOutputImageData[idx]); 
                }
            }
            printf("\n"); 
        }
    */
    cudaFree(deviceInputImageData); 
    cudaFree(deviceOutputImageData); 
    cudaFree(deviceMaskData); 

    gpuTKTime_stop(Copy, "Copying data from the GPU");

    gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    gpuTKSolution(arg, outputImage);
    // cout << outputImage << '\n'; 

    // free(hostInputImageData); maybe not supposed to free here idk
    // free(hostOutputImageData); 
    //@@ Insert code here
    free(hostMaskData);
    gpuTKImage_delete(outputImage);
    gpuTKImage_delete(inputImage);

    return 0;
}
