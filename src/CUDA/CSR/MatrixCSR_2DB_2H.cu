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
#define XBD 128
#define YBD 8
const dim3 BLOCK_DIM(XBD, YBD);


void CpuMatrixVector(int rows, int *IRP, int *JA, double *AS, double *x, double *y) {
    for (int i = 0; i < rows; i++) {
        double t = 0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            t += AS[j] * x[JA[j]];
        }
        y[i] = t;
    }
}


__device__ void rowReduce(volatile double *sdata, int tid, int s) {
    switch (s) {
        case 16:
            sdata[tid] += sdata[tid + 16];
        case 8:
            sdata[tid] += sdata[tid + 8];
        case 4:
            sdata[tid] += sdata[tid + 4];
        case 2:
            sdata[tid] += sdata[tid + 2];
        case 1:
            sdata[tid] += sdata[tid + 1];
    }
}

__global__ void gpuMatrixVector(int rows, int *IRP, int *JA, double *AS, double *x, double *y) {
    __shared__ double ax[YBD][XBD];
    int tr = threadIdx.y;
    int tc = threadIdx.x;
    int row = blockIdx.x * blockDim.y + tr;
    int s;
    ax[tr][tc] = 0.0;
    if (row < rows) {
        int idxm = IRP[row] + tc;
        int idxn = IRP[row + 1];
        double t1 = 0.0;
        double t2 = 0.0;
        for (int i = idxm; i < idxn; i += XBD * 2) {
            t1 += AS[i] * x[JA[i]];
            if (i + XBD < idxn) { t2 += AS[i + XBD] * x[JA[i + XBD]]; }
            idxm += XBD * 2;
        }
        ax[tr][tc] = t1 + t2;
    }
    __syncthreads();
    for (s = XBD / 2; s >= 32; s >>= 1) {
        if (tc < s) {
            ax[tr][tc] += ax[tr][tc + s];
        }
        __syncthreads();
    }
    s = min(16, XBD / 2);
    if (tc < s) {
        rowReduce(&(ax[tr][0]), tc, s);
    }
    if ((tc == 0) && (row < rows)) {
        y[row] = ax[tr][tc];
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
    CpuMatrixVector(nrows, h_IRP, h_JA, h_AS, h_x, h_y);

    timer->stop();
    double cpuflops = flopcnt / timer->getTime();
    double CPUtime = timer->getTime();
//    std::cout << "  CPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << std::endl;

// ------------------------ Calculations on the GPU ------------------------- //

    // Calculate the dimension of the grid of blocks (1D) necessary to cover
    // all rows.
    //use 1 block per row
    const dim3 GRID_DIM((nrows - 1 + BLOCK_DIM.y) / BLOCK_DIM.y, 1);

    //print block size and grid size
//    printf("Block size: %d x %d\n", BLOCK_DIM.x, BLOCK_DIM.y);
//    printf("Grid size: %d x %d\n", GRID_DIM.x, GRID_DIM.y);

    timer->reset();
    timer->start();
    gpuMatrixVector<<<GRID_DIM, BLOCK_DIM >>>(nrows, d_IRP, d_JA, d_AS, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();
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
//        if (std::abs(h_y[row] - h_y_d[row]) > 1e-6)
//            printf("h_y[%d] = %f, h_y_d[%d] = %f, diff = %f\n", row, h_y[row], row, h_y_d[row],
//                   std::abs(h_y[row] - h_y_d[row]));
    }
//    std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;
    // Rel diff should be as close as possible to unit roundoff; double
    // corresponds to IEEE single precision, so unit roundoff is
    // 1.19e-07
    //
    printf("NAME: %-15s TYPE: %-15s OPTION: %-15s CPU_TIME: %-15f GPU_TIME: %-15f CPU_GFLOPS: %-15f GPU_GFLOPS: %-15f MAX_DIFF: %-15f MAX_REL_DIFF: %-15f SPEEDUP: %-15f \n",
           argv[0], argv[1], "CSR", CPUtime, GPUtime, cpuflops, gpuflops, diff, reldiff, CPUtime / GPUtime);


// ------------------------------- Cleaning up ------------------------------ //

    delete timer;

    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));

//    delete[] h_y_d;
//    delete[] h_IRP;
//    delete[] h_JA;
//    delete[] h_AS;
//    delete[] h_x;
//    delete[] h_y;

    return 0;
}