// 
// Author: Salvatore Filippone salvatore.filippone@cranfield.ac.uk
//

// Computes matrix-vector product. Matrix A is in row-major order
// i.e. A[i, j] is stored in i * ncols + j element of the vector.
//

#include <iostream>
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers
#include <cstdio>
#include <cstdlib>
#include <utility>
#include "../../mmio.h"
#include "../../MatrixBase.h"
#include "../../OMP/MatrixCSR.h"
#include "../../wtime.h"

using namespace std;

inline double dmin(double a, double b) { return a < b ? a : b; }

//const int ntimes = 5;

// Simple 1-D thread block
// Size should be at least 1 warp
#define BD 256
const dim3 BLOCK_DIM(BD);

#define cudaCheckError(msg) { \
    cudaError_t__err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        printf("CUDA error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            printf("Aborting...\n");       \
                              exit(1);     \
    } while (0)

// GPU implementation of matrix_vector product using a block of threads for
// each row. 
__device__ void rowReduce(volatile double *sdata, int tid) {
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2];
    sdata[tid] += sdata[tid +  1];
}

__global__ void gpuMatrixVector(int rows, int *IRP, int *JA, double *AS, double *x, double *y) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    __shared__ double sdata[BD];
    int s = 0;
    sdata[tid] = 0;
    if (row < rows) {
        double sum = 0;
        int start = IRP[row] + tid;
        int end = IRP[row + 1];
        for (int i = start; i < end; i += BD) {
            sum += AS[i] * x[JA[i]];
        }
        sdata[tid] = sum;
    }
    __syncthreads();

    for (int i = BD / 2; i >= 32; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }

    s = min(16, BD / 2);

    if (tid < s) {
        rowReduce(sdata, tid);
    }

    if (tid == 0) {
        y[row] = sdata[0];
    }
}

int main(int argc, char** argv) {
    int nrows, ncols, nz;

    int *I, *J;
    double *val;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }

    MatrixBase::readFile(nrows, ncols, nz, I, J, val, argv[1]);

    MatrixBase::sortData(I, J, val, nz);

    double *h_x = new double[nrows];
    h_x = MatrixBase::generateVector(nrows);

    MatrixCSR csr(nrows, ncols, nz, I, J, val, h_x);
// ----------------------- Host memory initialisation ----------------------- //
    //  Allocate memory space on the host.
    int *h_IRP = new int[nrows + 1];
    int *h_JA = new int[nz];
    double *h_AS = new double[nz];
    double *h_y = new double[nz];
    double *h_y_d = new double[nz];

    //IRP
    h_IRP = csr.getIRP();
    //JA
    h_JA = csr.getJA();
    //AS
    h_AS = csr.getAS();

//    //print IRP
//    printf("IRP: ");
//    for (int i = 0; i < nrows + 1; i++) {
//        printf("%d ", h_IRP[i]);
//    }
//    printf("\n");
//
//    //print JA
//    printf("JA: ");
//    for (int i = 0; i < nz; i++) {
//        printf("%d ", h_JA[i]);
//    }
//    printf("\n");
//
//    //print AS
//    printf("AS: ");
//    for (int i = 0; i < nz; i++) {
//        printf("%f ", h_AS[i]);
//    }
//    printf("\n");
//
//    //print x
//    printf("x: ");
//    for (int i = 0; i < nrows; i++) {
//        printf("%f ", h_x[i]);
//    }
//    printf("\n");


// ---------------------- Device memory initialisation ---------------------- //
    //  Allocate memory space on the device.
    int *d_IRP, *d_JA;
    double *d_x, *d_y, *d_AS;

    // allocate memory for A, x and y
    checkCudaErrors(cudaMalloc((void **) &d_x, nrows * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_y, nz * sizeof(double)));

    // allocate memory for IRP, JA and AS
    checkCudaErrors(cudaMalloc((void **) &d_IRP, (nrows + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_JA, nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_AS, nz * sizeof(double)));

    // Copy matrices from the host (CPU) to the device (GPU).
    checkCudaErrors(cudaMemcpy(d_IRP, h_IRP, (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_JA, h_JA, nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_AS, h_AS, nz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x, nrows * sizeof(double), cudaMemcpyHostToDevice));

    // ------------------------ Calculations on the CPU ------------------------- //
    double flopcnt = 2.e-6 * nz;

    // Create the CUDA SDK timer.
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);

    timer->start();
    MatrixBase::CSR_CpuMatrixVector(nrows, h_IRP, h_JA, h_AS, h_x, h_y);

    timer->stop();
    double cpuflops = flopcnt / timer->getTime();
    double CPUtime = timer->getTime();
//    std::cout << "  CPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << std::endl;

// ------------------------ Calculations on the GPU ------------------------- //

    // Calculate the dimension of the grid of blocks (1D) necessary to cover
    // all rows.
    //use 1 block per row
    const dim3 GRID_DIM((nrows - 1 + BLOCK_DIM.y) / BLOCK_DIM.y, 1, 1);

    timer->reset();
    timer->start();
    gpuMatrixVector<<<GRID_DIM, BLOCK_DIM >>>(nrows, d_IRP, d_JA, d_AS, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();

    //get errors from kernel
    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess)
        printf("Error name: %s \t Error description: %s \t Error code: %d \t \n", cudaGetErrorName(err),
               cudaGetErrorString(err), err);

    double gpuflops = flopcnt / timer->getTime();
    double GPUtime = timer->getTime();
//    std::cout << "  GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops << std::endl;

    // Download the resulting vector d_y from the device and store it in h_y_d.
    checkCudaErrors(cudaMemcpy(h_y_d, d_y, nrows * sizeof(double), cudaMemcpyDeviceToHost));

//    printf("CPU result \t GPU result \t Difference\n");
//    for (int i = 0; i < nrows; i++) {
//        if(h_y[i] != h_y_d[i])
//            printf("%f \t %f \t %f\n", h_y[i], h_y_d[i], h_y[i] - h_y_d[i]);
//    }

    // Now let's check if the results are the same.
    double reldiff = 0.0f;
    double diff = 0.0f;

    for (int row = 0; row < nrows; ++row) {
        double maxabs = std::max(std::abs(h_y[row]), std::abs(h_y_d[row]));
        if (maxabs == 0.0) maxabs = 1.0;
        reldiff = std::max(reldiff, std::abs(h_y[row] - h_y_d[row]) / maxabs);
        diff = std::max(diff, std::abs(h_y[row] - h_y_d[row]));
//        if (std::abs(h_y[row] - h_y_d[row]) != 0.0)
//            printf("h_y[%d] = %f, h_y_d[%d] = %f, diff = %f\n", row, h_y[row], row, h_y_d[row],
//                   std::abs(h_y[row] - h_y_d[row]));
    }
    printf("NAME: %-15s CPU_TIME: %-10f  GPU_TIME: %-10f  CPU_GFLOPS: %-10f  GPU_GFLOPS: %-10f  MAX_DIFF: %-10e  MAX_REL_DIFF: %-10e\n",
           argv[0], CPUtime, GPUtime, cpuflops, gpuflops, diff, reldiff);

    // Rel diff should be as close as possible to unit roundoff; double
    // corresponds to IEEE single precision, so unit roundoff is
    // 1.19e-07
    //

// ------------------------------- Cleaning up ------------------------------ //

    delete timer;

    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));

    delete[] h_y_d;
    delete[] h_IRP;
    delete[] h_JA;
    delete[] h_AS;
    delete[] h_x;
    delete[] h_y;

    return 0;
}