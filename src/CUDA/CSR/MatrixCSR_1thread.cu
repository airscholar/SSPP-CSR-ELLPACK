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
#include <map>
#include "../../mmio.h"
#include "../../MatrixBase.h"
#include "../../OMP/MatrixCSR.h"

#include "../../wtime.h"

using namespace std;

inline double dmin(double a, double b) { return a < b ? a : b; }

//const int ntimes = 5;

//Simple dimension: define a 1D block structure
#define BD 256
const dim3 BLOCK_DIM(BD);

void CpuMatrixVector(int rows, int *IRP, int *JA, double *AS, double *x, double *y) {
    for (int i = 0; i < rows; i++) {
        double t = 0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            t += AS[j] * x[JA[j]];
        }
        y[i] = t;
    }
}

// GPU implementation of matrix_vector product using a block of threads for
// each row.
__device__ void rowReduce(volatile double *sdata, int tid) {
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// GPU implementation of matrix_vector product: see if you can use
// one thread per row. You'll need to get the addressing right!
// each block of rows.
__global__ void gpuMatrixVector(int rows, int *IRP, int *JA, double *AS, double *x, double *y) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    __shared__ double sdata[BD];
    sdata[tid] = 0;
    int s;
    if (row < rows) {
        double sum = 0;
        int start = IRP[row];
        int end = IRP[row + 1];
        for (int i = start + tid; i < end; i += BD) {
            sum += AS[i] * x[JA[i]];
        }
        sdata[tid] = sum;
    }
    __syncthreads();

    for (int i = blockDim.x / 2; i >= 32; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }

    s = min(16, blockDim.x / 2);

    if (tid < s) {
        rowReduce(sdata, tid);
    }

    __syncthreads();

    if (tid == 0) {
        if (row < rows) {
            y[row] = sdata[0];
        }
    }
}


int main(int argc, char **argv) {
    int nrows, ncols, nz;
    int ret_code;
    MM_typecode matcode;

    int *I, *J;
    double *val;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }

    MatrixBase::readFile(nrows, ncols, nz, I, J, val, argv[1]);


    MatrixBase::sortData(I, J, val, nz);

    double *h_x = new double[nrows];
    generateVector(nrows, h_x);

    MatrixCSR csr(nrows, ncols, nz, I, J, val, h_x);
// ----------------------- Host memory initialisation ----------------------- //
    //  Allocate memory space on the host.
    int *h_IRP = new int[nrows + 1];
    int *h_JA = new int[nz];

    h_IRP = csr.getIRP();
    h_JA = csr.getJA();

    //convert double to double
    double *h_AS = new double[nz];
    h_AS = csr.getAS();

    double *h_y = new double[nz];
    double *h_y_d = new double[nz];

    // vector X is initialised to 1
//    std::cout << "Matrix-vector product: single thread per row version " << std::endl;
//    std::cout << "Test case: " << nrows << " x " << ncols << std::endl;

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
    const dim3 GRID_DIM((nrows - 1 + BLOCK_DIM.x) / BLOCK_DIM.x, 1);

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

//    printf("CPU result \t GPU result \t Difference \t Relative difference");
//    for (int i = 0; i < nrows; i++) {
//        printf("%f \t %f \t %f \t %f\n", h_y[i], h_y_d[i], h_y[i] - h_y_d[i], (h_y[i] - h_y_d[i]) / h_y[i]);
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
    printf("NAME: %-15s TYPE: %-15s OPTION: %-15s CPU_TIME: %-15f GPU_TIME: %-15f CPU_GFLOPS: %-15f GPU_GFLOPS: %-15f MAX_DIFF: %-15f MAX_REL_DIFF: %-15f SPEEDUP: %-15f \n",
           argv[0], argv[1], "CSR", CPUtime, GPUtime, cpuflops, gpuflops, diff, reldiff, CPUtime / GPUtime);

//    std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;
    // Rel diff should be as close as possible to unit roundoff; double
    // corresponds to IEEE single precision, so unit roundoff is
    // 1.19e-07
    //

// ------------------------------- Cleaning up ------------------------------ //

    delete timer;

//    checkCudaErrors(cudaFree(d_x));
//    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));

    delete[] h_IRP;
    delete[] h_JA;
    delete[] h_AS;
//    delete[] h_x;
//    delete[] h_y;

    return 0;
}