#include <cstdio>
#include <cstdlib>
#include "../../OMP_Tests/MatrixCSR_Test.h"
#include <utility>
#include "../../MatrixBase.h"
#include "../MatrixELLPACK.h"
#include "../../wtime.h"
#include <map>
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
        tmlt += t2 - t1;
    }
    serialGflops = (2.0 * nz / (tmlt / ntimes) * 1e-6) * 0.001;

#pragma omp parallel
    {
#pragma omp master
        printf("NAME: %s, TYPE: %s, NUMPROCS: %d, SERIAL: %f, UNROLL1V_TIME: %f, UNROLL1V_GFLOPS: %f, UNROLL1V_MAX_ERROR: %f, UNROLL1H_TIME: %f, UNROLL1H_GFLOPS: %f, UNROLL2H_MAX_ERROR: %f\n", argv[1],
               "ELL", omp_get_num_threads(), serialGflops, unrollVTime, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
//    delete[] x;
//    delete[] y;
    delete[] I;
    delete[] J;
    delete[] val;

    return 0;
}