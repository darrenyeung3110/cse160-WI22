#include <gputk.h>

// Compute C = A anticlockwise rotation
// Sgemm stands for single precision general matrix-matrix multiply
__global__ void sgemm(float *A, float *C, int numARows,
                      int numAColumns) {
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if(row<numAColumns && col<numARows) {
    int mappedRow = col;
    int mappedColumn = numAColumns - row -1;
    C[row * numARows + col] = A[mappedRow*numAColumns + mappedColumn];
  }
}

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceC;
  int numARows;    // number of rows in the matrix
  int numAColumns; // number of columns in the matrix

  int numCRows;
  int numCColumns;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numARows * numAColumns * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  numCRows    = numAColumns;
  numCColumns = numARows;

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  gpuTKCheck(cudaMalloc((void **)&deviceA,
                     numARows * numAColumns * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&deviceC,
                     numARows * numAColumns * sizeof(float)));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  gpuTKCheck(cudaMemcpy(deviceA, hostA,
                     numARows * numAColumns * sizeof(float),
                     cudaMemcpyHostToDevice));

  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(2, 2);
// changed to BColumns and ARows from Acolumns and BRows
  dim3 gridDim(ceil(((float)numARows) / blockDim.x),
               ceil(((float)numAColumns) / blockDim.y));

  gpuTKLog(TRACE, "The block dimensions are ", blockDim.x, " x ", blockDim.y);
  gpuTKLog(TRACE, "The grid dimensions are ", gridDim.x, " x ", gridDim.y);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  gpuTKCheck(cudaMemset(deviceC, 0, numARows * numAColumns * sizeof(float)));
  sgemm<<<gridDim, blockDim>>>(deviceA, deviceC, numARows,
                               numAColumns);
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  gpuTKCheck(cudaMemcpy(hostC, deviceC,
                     numARows * numAColumns * sizeof(float),
                     cudaMemcpyDeviceToHost));
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  for (int i=0;i<numCRows;i++){
        for (int j=0;j<numCColumns;j++){
            printf("%f ", hostC[i*numCColumns + j]);
        }
        printf("\n");
    }
  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceC);
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  // gpuTKSolution(args, hostC, numARows, numBColumns);

  free(hostA);
  free(hostC);

  return 0;
}
