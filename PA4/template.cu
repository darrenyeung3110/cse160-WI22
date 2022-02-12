#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLOCK_WIDTH 16


// TODO: WHEN DIMENSIONS ARE NOT A MULTIPLE OF 16, SHIT IS WRONG

// Compute C = A * B
__device__ int minimum(int a, int b) {
    if (a < b ) {
        return a; 
    } else {
        return b; 
    }
}
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this lab
    __shared__ float subMatrixA[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float subMatrixB[BLOCK_WIDTH][BLOCK_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y; 

    // row, col refers to the element that we are working on in the output matrix
    int row = (by * BLOCK_WIDTH) + ty;
    int col = (bx * BLOCK_WIDTH) + tx;

    float dotProduct = 0; 
    int coverLimit = ceil(((float)numAColumns) / BLOCK_WIDTH) * BLOCK_WIDTH - 1; 
    int elementsLeft = numAColumns; // also = to numBRows
    for (int i = tx, j = ty, cover = BLOCK_WIDTH-1; cover <= coverLimit; i += BLOCK_WIDTH, j += BLOCK_WIDTH, cover += BLOCK_WIDTH) {
        int aRow = row; int aCol = i; 
        int bRow = j; int bCol = col; 
        // bring in a value from the A matrix into shared memory
        if (aRow < numARows && aCol < numAColumns) {
            int collapsedAIndex = (aRow * numAColumns) + aCol; 
            float aValue = A[collapsedAIndex]; 
            subMatrixA[ty][tx] = aValue; 
        }
        // bring in a value from the B matrix into shared memory
        if (bRow < numBRows && bCol < numBColumns) {
            int collapsedBIndex = (bRow * numBColumns) + bCol; 
            float bValue = B[collapsedBIndex]; 
            subMatrixB[ty][tx] = bValue; 
        }
        __syncthreads(); 
        // computing a partial dot product 
        if ((row <  numCRows && col < numCColumns)) {
            int limit = minimum(BLOCK_WIDTH, elementsLeft); 
            for (int k = 0; k < limit; k++) {
                dotProduct += subMatrixA[ty][k] * subMatrixB[k][tx]; 
            }
            elementsLeft -= limit; 
        }
        __syncthreads(); 
    }
    // if row and and col is in range of the C matrix
    if ((row <  numCRows && col < numCColumns)) {
        C[row*numCColumns+col] = dotProduct; 
    }
}

int main(int argc, char **argv) {
    gpuTKArg_t args;
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostBTransposed;
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;    // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
    int numBTransposedRows;
    int numBTransposedColumns;

    args = gpuTKArg_read(argc, argv);

    gpuTKTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
    hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);

    int aSize = numARows * numAColumns;
    int bSize = numBRows * numBColumns;
    //@@ Set numCRows and numCColumns
    numCRows = numARows; 
    numCColumns = numBColumns; 
    int cSize = numCRows * numCColumns; 

    //@@ Allocate the hostC matrix 
    hostC = (float*)malloc(cSize * sizeof(float));
    //@@ Allocate the hostBTransposed matrix
    hostBTransposed = (float*)malloc(bSize * sizeof(float));
    gpuTKTime_stop(Generic, "Importing data and creating memory on host");

    gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    gpuTKTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void**)&deviceA, aSize * (sizeof(float)));
    cudaMalloc((void**)&deviceB, bSize * (sizeof(float)));
    cudaMalloc((void**)&deviceC, cSize * (sizeof(float)));
    gpuTKTime_stop(GPU, "Allocating GPU memory.");

    //@@ Transpose B matrix here
    /*
        for (int i = 0; i < numBRows; i++) {
            for (int j = 0; j < numBColumns; j++) {
                int bti = j; int btj = i; 
                int rawIndexOriginal = i * numBColumns + j; 
                int rawIndexTransposed = bti * numBTransposedColumns + btj;
                hostBTransposed[rawIndexTransposed] = hostB[rawIndexOriginal]; 
            }
        }
    */

    gpuTKTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, aSize*(sizeof(float)), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, bSize*(sizeof(float)), cudaMemcpyHostToDevice);
    gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    int gridRows = ceil(((float)numCRows) / BLOCK_WIDTH); 
    int gridColumns = ceil(((float)numCColumns) / BLOCK_WIDTH); 

    dim3 gridSize(gridColumns, gridRows); 
    dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH); 

    gpuTKTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows,
    numBColumns, numCRows, numCColumns);
    cudaDeviceSynchronize();
    gpuTKTime_stop(Compute, "Performing CUDA computation");

    gpuTKTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, cSize *(sizeof(float)), cudaMemcpyDeviceToHost);
    gpuTKTime_stop(Copy, "Copying output memory to the CPU");

    gpuTKTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA); 
    cudaFree(deviceB); 
    cudaFree(deviceC); 

    gpuTKTime_stop(GPU, "Freeing GPU Memory");

    gpuTKSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

