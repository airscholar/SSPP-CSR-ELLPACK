#include <cstdio>
#include <cstdlib>
#include "../../OMP_Tests/MatrixCSR_Test.h"
#include <utility>
#include "../../MatrixBase.h"
#include "../MatrixCSR.h"
#include "../MatrixELLPACK.h"
#include "../../wtime.h"
#include <map>
#include <omp.h>
#include <string>
#include "../../SparseMatrix.h"

using namespace std;

inline double dmin(double a, double b) { return a < b ? a : b; }

const int ntimes = 5;

int main(int argc, char *argv[]) {
    int M, N, nz;
    int *I, *J;
    double *val;

    //THIS IS FOR UNIT TESTING PURPOSES ONLY
//    MatrixCSR_Test matrixCSR_test(M, N, nz, I, J, val);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }

    MatrixBase::readFile(M, N, nz, I, J, val, argv[1]);

    MatrixBase::sortData(I, J, val, nz);

    double *x = MatrixBase::generateVector(M);
    double *y = new double[M];

    MatrixCSR matrixCSR(M, N, nz, I, J, val, x);
    double tmlt = 1e100;
    //gflops
    double serialGflops = 0, unrollVGflops = 0, unrollHGflops = 0;
    double *serialResult = new double[M];
    double *unrollVResult = new double[M];
    double *unrollHResult = new double[M];
    double serialTime = 0, unrollVTime = 0, unrollHTime = 0;

    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        serialResult = matrixCSR.serialMultiply(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    serialGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        unrollVResult = matrixCSR.openMPMultiplyUnroll16V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    unrollVGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;

    for (int tr = 0; tr < M; tr++) {
        double t1 = wtime();
        unrollHResult = matrixCSR.openMPMultiplyUnroll16H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    unrollHGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

//    Unroll Vertical Max Error
    float diff = 0;
    for (int i = 0; i < M; i++) {
        float err = serialResult[i] - unrollVResult[i];
        if (err < 0) err = -err;
        if (err > diff) diff = err;
    }

    //    Unroll Horizontal Max Error
    float diff2 = 0;
    for (int i = 0; i < M; i++) {
        float err = serialResult[i] - unrollHResult[i];
        if (err < 0) err = -err;
        if (err > diff2) diff2 = err;
    }
#pragma omp parallel
    {
#pragma omp master
        printf("NAME: %s, TYPE: %s, NUMPROCS: %d, SERIAL: %f, UNROLL16V_TIME: %f, UNROLL16V_GFLOPS: %f, UNROLL16V_MAX_ERROR: %f, UNROLL16H_TIME: %f, UNROLL16H_GFLOPS: %f, UNROLL16H_MAX_ERROR: %f\n", argv[1],
               "CSR", omp_get_num_threads(), serialGflops, unrollVTime, unrollVGflops, diff, unrollHTime, unrollHGflops, diff2);
    }

//    delete[] x;
//    delete[] y;
    delete[] I;
    delete[] J;
    delete[] val;

    return 0;
}