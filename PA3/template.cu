
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

#define BLOCK_WIDTH 2
// block size 2x2
// Compute C = A * Bx 
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
    int row = (blockIdx.x * BLOCK_WIDTH) + threadIdx.x; 
    int col  = (blockIdx.y * BLOCK_WIDTH) + threadIdx.y; 
    if (row >= 0 && row < numCRows && col >= 0  && col < numCColumns) {
      // printf("%d lol %d\n", row, col);
      int aPtr = row * numAColumns; 
      int bPtr = col; 
      float sum = 0.0; 
      for (int i = 0; i < numAColumns; i++) {
          float currentMultiply = A[aPtr] * B[bPtr]; 
          sum += currentMultiply; 
          aPtr++; 
          bPtr += numBColumns; // ?
      }
      int collapsedOutputIndex = (row * numCColumns) + col;
      C[collapsedOutputIndex] = sum; 
    }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
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


  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  int aSize = numARows * numAColumns; 
  int bSize = numBRows * numBColumns; 
  //@@ Set numCRows and numCColumns
  // ARowsXACols * BRowsXBCols = ARows X BCols
  numCRows    = numARows; 
  numCColumns = numBColumns; 
  int cSize = numCRows * numCColumns;
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(cSize * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA, aSize * (sizeof(float)));
  cudaMalloc((void**)&deviceB, bSize * (sizeof(float)));
  cudaMalloc((void**)&deviceC, cSize * (sizeof(float)));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, aSize*(sizeof(float)), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, bSize*(sizeof(float)), cudaMemcpyHostToDevice);
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int gridRows;
  int gridCols; 
  if (numCRows % 2 != 0) 
      gridRows = (numCRows + 1) / 2; 
  else 
      gridRows = (numCRows) / 2; 
  if (numCColumns % 2 != 0) 
      gridCols = (numCColumns + 1) / 2; 
  else 
      gridCols = numCColumns / 2; 

  dim3 grid_size(gridRows, gridCols); 
  dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH); // 2x2

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<grid_size, block_size>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, 
  numCRows, numCColumns);
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, cSize *(sizeof(float)), cudaMemcpyDeviceToHost); 
  // printf("%d\n", deviceC[0]);
  // printf("%d\n", hostC[0]);
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
