#include <cstdio>
#include <cstdlib>
#include "../../OMP_Tests/MatrixCSR_Test.h"
#include <utility>
#include "../../MatrixBase.h"
#include "../MatrixCSR.h"
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

    //THIS IS FOR UNIT TESTING PURPOSES ONLY
//    MatrixCSR_Test matrixCSR_test(M, N, nz, I, J, val);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }

    //read file
    MatrixBase::readFile(M, N, nz, I, J, val, argv[1]);

    //sort the data
    MatrixBase::sortData(I, J, val, nz);

    //generate vector x
    double *x = MatrixBase::generateVector(M);

    //CONVERSION TO CSR
    MatrixCSR matrixCSR(M, N, nz, I, J, val, x);

    double *y = new double[M];
    double tmlt = 1e100;
    double serialGflops = 0;
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
    serialTime = tmlt;
    serialGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

#pragma omp parallel
    {
#pragma omp master
        printf("NAME: %s, TYPE: %s, NUMPROCS: %d, SERIAL: %f, UNROLL1V_TIME: %f, UNROLL1V_GFLOPS: %f, UNROLL1V_MAX_ERROR: %f, UNROLL1H_TIME: %f, UNROLL1H_GFLOPS: %f, UNROLL2H_MAX_ERROR: %f\n", argv[1],
               "CSR", omp_get_num_threads(), serialGflops, unrollVTime, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

//    delete[] x;
//    delete[] y;
    delete[] I;
    delete[] J;
    delete[] val;

    return 0;
}