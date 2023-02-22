#include <iostream>
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers
#include <cstdio>
#include <cstdlib>
#include <utility>
#include "../../mmio.h"
#include "../../MatrixBase.h"

#include "../../OMP/MatrixELLPACK.h"
#include "../../wtime.h"

using namespace std;

inline double dmin(double a, double b) { return a < b ? a : b; }

//const int ntimes = 5;

// Simple 1-D thread block
// Size should be at least 1 warp
#define BD 32
const dim3 BLOCK_DIM(BD);

// GPU implementation of matrix_vector product using a block of threads for
// each row.
__device__ void rowReduce(volatile double *sdata, int tid) {
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// Simple CPU implementation of matrix addition.
void CpuMatrixVector(int rows, int *JA, double *AS, int maxNZ, double *x, double *y) {
    double t;
    int i, j;
    for (i = 0; i < rows; i++) {
        t = 0;
        for (j = 0; j < maxNZ; j++) {
            int index = i * maxNZ + j;
            if (index < rows * maxNZ && JA[index] < rows)
                t += AS[index] * x[JA[index]];
        }
        y[i] = t;
    }
}

__global__ void gpuMatrixVector(int rows, int *JA, double *AS, int maxNZ, double *x, double *y) {
    __shared__ double aux[BD];
    int tc = threadIdx.x;
    int row = blockIdx.x;
    aux[tc] = 0.0;
    int s = 0;
    if (row < rows) {
        double sum = 0.0;
        for (int j = 0; j < maxNZ; j += BD * 4) {
            int index1 = row * maxNZ + j + tc;
            int index2 = row * maxNZ + j + tc + BD;
            int index3 = row * maxNZ + j + tc + BD * 2;
            int index4 = row * maxNZ + j + tc + BD * 3;
            if (j + tc < maxNZ && index1 < rows * maxNZ) sum += AS[index1] * x[JA[index1]];
            if (j + tc + BD < maxNZ && index2 < rows * maxNZ) sum += AS[index2] * x[JA[index2]];
            if (j + tc + BD * 2 < maxNZ && index3 < rows * maxNZ) sum += AS[index3] * x[JA[index3]];
            if (j + tc + BD * 3 < maxNZ && index4 < rows * maxNZ) sum += AS[index4] * x[JA[index4]];
        }
        aux[tc] = sum;
    }
    __syncthreads();

    for (int i = BD / 2; i >= 32; i >>= 1) {
        if (tc < i) {
            aux[tc] += aux[tc + i];
        }
        __syncthreads();
    }

    s = min(16, BD / 2);

    if (tc < s) {
        rowReduce(aux, tc);
    }

    if (tc == 0) {
        y[row] = aux[0];
    }
}


int main(int argc, char **argv) {
    int nrows, ncols, nz;

    int *I, *J;
    double *val;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }

    // read in the matrix
    MatrixBase::readFile(nrows, ncols, nz, I, J, val, argv[1]);

    // sort the matrix
    MatrixBase::sortData(I, J, val, nz);

    // generate a random vector
    double *temp_x = new double[nrows];
    temp_x = MatrixBase::generateVector(nrows);

    // create an ELLPACK matrix
    MatrixELLPACK ellpack(nrows, ncols, nz, I, J, val, temp_x);

    // get the number of non-zero elements per row
    int maxNZ = ellpack.getMaxNZ();

//----------------------- transpose the matrix ----------------------- //
    //transpose JA
    int *h_JA = new int[nrows * maxNZ];
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < maxNZ; j++) {
            h_JA[j * nrows + i] = ellpack.getJA()[i * maxNZ + j];
        }
    }

    // transpose AS
    double *h_AS = new double[nrows * maxNZ];
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < maxNZ; j++) {
            h_AS[j * nrows + i] = ellpack.getAS()[i * maxNZ + j];
        }
    }

    //update nrows, ncols, nz to reflect the transpose
    int temp = nrows;
    nrows = maxNZ;
    maxNZ = temp;

// ----------------------- Host memory initialisation ----------------------- //
    //  Allocate memory space on the host.
    double *h_x = new double[nrows];
    double *h_y = new double[nz];
    double *h_y_d = new double[nz];

    h_x = MatrixBase::generateVector(nrows);

// ---------------------- Device memory initialisation ---------------------- //
    //  Allocate memory space on the device.
    int *d_JA;
    double *d_x, *d_y, *d_AS;

    // allocate memory for x and y
    checkCudaErrors(cudaMalloc((void **) &d_x, nrows * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_y, nz * sizeof(double)));

    // allocate memory for JA and AS
    checkCudaErrors(cudaMalloc((void **) &d_JA, nrows * maxNZ * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_AS, nrows * maxNZ * sizeof(double)));

    // Copy matrices from the host (CPU) to the device (GPU).
    checkCudaErrors(cudaMemcpy(d_JA, h_JA, nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_AS, h_AS, nz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x, nrows * sizeof(double), cudaMemcpyHostToDevice));

    // ------------------------ Calculations on the CPU ------------------------- //
    double flopcnt = 2.e-6 * nz;

    // Create the CUDA SDK timer.
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);

    timer->start();
    CpuMatrixVector(nrows, h_JA, h_AS, maxNZ, h_x, h_y);

    timer->stop();
    double cpuflops = flopcnt / timer->getTime();
    double CPUtime = timer->getTime();

// ------------------------ Calculations on the GPU ------------------------- //
    // Calculate the dimension of the grid of blocks (1D) necessary to cover all rows.
    // If the matrix size is less than the block size, then only one block is needed.
    const dim3 GRID_DIM(nrows, 1);

    //print block size and grid size
//    printf("Block size: %d x %d\n", BLOCK_DIM.x, BLOCK_DIM.y);
//    printf("Grid size: %d x %d\n", GRID_DIM.x, GRID_DIM.y);

    timer->reset();
    timer->start();

    gpuMatrixVector<<<GRID_DIM, BLOCK_DIM >>>(nrows, d_JA, d_AS, maxNZ, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
    double gpuflops = flopcnt / timer->getTime();
    double GPUtime = timer->getTime();

    // Download the resulting vector d_y from the device and store it in h_y_d.
    checkCudaErrors(cudaMemcpy(h_y_d, d_y, nrows * sizeof(double), cudaMemcpyDeviceToHost));

    // Now let's check if the results are the same.
    double reldiff = 0.0f;
    double diff = 0.0f;

    for (int row = 0; row < nrows; ++row) {
        double maxabs = std::max(std::abs(h_y[row]), std::abs(h_y_d[row]));
        if (maxabs == 0.0) maxabs = 1.0;
        reldiff = std::max(reldiff, std::abs(h_y[row] - h_y_d[row]) / maxabs);
        diff = std::max(diff, std::abs(h_y[row] - h_y_d[row]));
    }

    printf("NAME: %-15s TYPE: %-15s OPTION: %-15s CPU_TIME: %-15f GPU_TIME: %-15f CPU_GFLOPS: %-15f GPU_GFLOPS: %-15f MAX_DIFF: %-15f MAX_REL_DIFF: %-15f SPEEDUP: %-15f \n",
           argv[0], argv[1], "ellpack", CPUtime, GPUtime, cpuflops, gpuflops, diff, reldiff, CPUtime / GPUtime);

// ------------------------------- Cleaning up ------------------------------ //

    delete timer;

    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));

//    delete[] h_y_d;
//    delete[] h_JA;
//    delete[] h_AS;
//    delete[] h_x;
//    delete[] h_y;

    return 0;
}