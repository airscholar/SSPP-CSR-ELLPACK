#include <cstdio>
#include <cstdlib>
#include "../../OMP_Tests/MatrixCSR_Test.h"
#include <utility>
#include "../../MatrixBase.h"
#include "../MatrixELLPACK.h"
#include "../../wtime.h"
#include <omp.h>
#include <string>

using namespace std;

inline double dmin(double a, double b) { return a < b ? a : b; }

const int ntimes = 5;

int main(int argc, char *argv[]) {
    int M, N, nz;
    int *I, *J;
    double *val;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }

    MatrixBase::readFile(M, N, nz, I, J, val, argv[1]);

    MatrixBase::sortData(I, J, val, nz);

    double *x = MatrixBase::generateVector(M);
    double *y = new double[M];

    MatrixELLPACK ellpack(M, N, nz, I, J, val, x);

    double tmlt = 1e100;
    //gflops
    double serialGflops = 0, unrollVGflops = 0, unrollHGflops = 0;
    double *serialResult = new double[M];
    double *unrollVResult = new double[M];
    double *unrollHResult = new double[M];
    double serialTime = 0, unrollVTime = 0, unrollHTime = 0;

    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        serialResult = ellpack.serialMultiply(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    serialTime = tmlt;
    serialGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        unrollVResult = ellpack.openMPMultiplyUnroll2V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    unrollVTime = tmlt;
    unrollVGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;

    for (int tr = 0; tr < M; tr++) {
        double t1 = wtime();
        unrollHResult = ellpack.openMPMultiplyUnroll2H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    unrollHTime = tmlt;
    unrollHGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

//    Unroll Vertical Max Error
    double diff = MatrixBase::compute_Max_Error(serialResult, unrollVResult, M);
    //    Unroll Horizontal Max Error
    double diff2 = MatrixBase::compute_Max_Error(serialResult, unrollHResult, M);

#pragma omp parallel
    {
#pragma omp master
        printf("NAME: %s, TYPE: %s, NUMPROCS: %d, SERIAL: %f, UNROLL2V_TIME: %f, UNROLL2V_GFLOPS: %f, UNROLL2V_MAX_ERROR: %f, UNROLL2H_TIME: %f, UNROLL2H_GFLOPS: %f, UNROLL2H_MAX_ERROR: %f\n",
               argv[1],
               "ELL", omp_get_num_threads(), serialGflops, unrollVTime, unrollVGflops, diff, unrollHTime, unrollHGflops,
               diff2);
    }
//    delete[] x;
//    delete[] y;
    delete[] I;
    delete[] J;
    delete[] val;

    return 0;
}