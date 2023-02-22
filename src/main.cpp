#include <cstdio>
#include <cstdlib>
#include "OMP_Tests/MatrixCSR_Test.h"
#include <utility>
#include "MatrixBase.h"
#include "OMP/MatrixCSR.h"
#include "OMP/MatrixELLPACK.h"
#include "wtime.h"
#include <map>
#include <omp.h>
#include <string>
#include "SparseMatrix.h"

using namespace std;

inline double dmin(double a, double b) { return a < b ? a : b; }

map<pair<int, int>, double> matrix; // use a map to store the values of I, J and V
const int ntimes = 5;

void
printFormattedResults(char *fileName, std::string matrixType, double serialGflops, double u2VGflops, double u2HGflops,
                      double u4VGflops, double u4HGflops, double u8VGflops, double u8HGflops,
                      double u16VGflops, double u16HGflops) {
    // print results
    printf("%-30s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n", "FileName", "Matrix", "NumProcs", "Serial",
           "Unroll2V", "Unroll2H",
           "Unroll4V", "Unroll4H", "Unroll8V", "Unroll8H", "Unroll16V", "Unroll16H");
    //print gflops
#pragma omp parallel
    {
#pragma omp master
        printf("%-30s %-10s %-10d %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f\n", fileName,
               matrixType.c_str(),
               omp_get_num_threads(),
               serialGflops, u2VGflops, u2HGflops, u4VGflops, u4HGflops, u8VGflops, u8HGflops, u16VGflops, u16HGflops);
    }
}

void CSRResult(char *fileName, std::string matrixType, double *x, double *y, int M, int nz, MatrixCSR matrixCSR) {
    double tmlt = 1e100;
    //gflops
    double serialGflops = 0, u2VGflops = 0, u2HGflops = 0, u4VGflops = 0, u8VGflops = 0, u16VGflops = 0, u4HGflops = 0, u8HGflops = 0, u16HGflops = 0;

    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixCSR.serialMultiply(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    serialGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixCSR.openMPMultiplyUnroll2V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u2VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;

    for (int tr = 0; tr < M; tr++) {
        double t1 = wtime();
        matrixCSR.openMPMultiplyUnroll2H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u2HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixCSR.openMPMultiplyUnroll4H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u4HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixCSR.openMPMultiplyUnroll4V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u4VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixCSR.openMPMultiplyUnroll8V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u8VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixCSR.openMPMultiplyUnroll8H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u8HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixCSR.openMPMultiplyUnroll16H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u16HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixCSR.openMPMultiplyUnroll16V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u16VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    printFormattedResults(fileName, matrixType, serialGflops, u2VGflops, u2HGflops, u4VGflops, u4HGflops, u8VGflops,
                          u8HGflops,
                          u16VGflops, u16HGflops);
}

void ELLPACKResult(char *fileName, std::string matrixType, double *x, double *y, int M, int nz,
                   MatrixELLPACK matrixELLPack) {
    double tmlt = 1e100;
    //gflops
    double serialGflops = 0, u2VGflops = 0, u2HGflops = 0, u4VGflops = 0, u8VGflops = 0, u16VGflops = 0, u4HGflops = 0, u8HGflops = 0, u16HGflops = 0;

    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixELLPack.serialMultiply(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    serialGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixELLPack.openMPMultiplyUnroll2V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u2VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < M; tr++) {
        double t1 = wtime();
        matrixELLPack.openMPMultiplyUnroll2H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u2HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixELLPack.openMPMultiplyUnroll4H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u4HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixELLPack.openMPMultiplyUnroll4V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u4VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixELLPack.openMPMultiplyUnroll8V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u8VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixELLPack.openMPMultiplyUnroll8H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u8HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixELLPack.openMPMultiplyUnroll16H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u16HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        matrixELLPack.openMPMultiplyUnroll16V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    u16VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    printFormattedResults(fileName, matrixType, serialGflops, u2VGflops, u2HGflops, u4VGflops, u4HGflops, u8VGflops,
                          u8HGflops,
                          u16VGflops, u16HGflops);
}

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
    CSRResult(argv[1], "CSR", x, y, M, nz, matrixCSR);

//// multiply
//    double tmlt = 1e100;
//    double *serialCSRResult = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        double t1 = wtime();
//        serialCSRResult = matrixCSR.serialMultiply(x, y);
//        double t2 = wtime();
//        tmlt = dmin(tmlt, (t2 - t1));
//    }
//    printf("Serial Done\n");
//    tmlt = 1e100;
//    double *parallelResult = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        double t1 = wtime();
//        parallelResult = matrixCSR.openMPMultiplyUnroll16H(x, y);
//        double t2 = wtime();
//        tmlt = dmin(tmlt, (t2 - t1));
//    }
//
////    validate result
//    double diff = 0;
//    for (int i = 0; i < M; i++) {
//        double err = serialCSRResult[i] - parallelResult[i];
//        if (err < 0) err = -err;
//        diff += err;
//    }
//    printf("Error: %f \n", diff);



//    printf("ELLPACK");
    MatrixELLPACK ellpack(M, N, nz, I, J, val, x);
    ELLPACKResult(argv[1], "ELLPACK", x, y, M, nz, ellpack);
//
//    double* y1 = new double[M];
//    double tmlt1 = 1e100;
//    double *serialEllPackResult = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        long double t1 = wtime();
//        serialEllPackResult = ellpack.serialMultiply(x, y1);
//        double t2 = wtime();
//        tmlt1 = dmin(tmlt1, (t2 - t1));
//    }
//    double gflops1 = (2.0 * nz / tmlt1 * 1e-6) * 0.001;
//    printf("Serial ELLPACK %d x %d: time %lf  GFLOPS: %f \n", M, N, tmlt1, gflops1);
//
//    tmlt1 = 1e100;
//    double *u2Hresult = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        double t1 = wtime();
//        u2Hresult = ellpack.openMPMultiplyUnroll16V(x, y);
//        double t2 = wtime();
//        tmlt1 = dmin(tmlt1, (t2 - t1));
//    }
//    gflops1 = (2.0 * nz / tmlt1 * 1e-6) * 0.001;
//    printf("OMP ELLPACK %d x %d: time %lf  GFLOPS: %f \n", M, N, tmlt1, gflops1);
//
//    //validate result
//    double diff1 = 0;
//    for (int i = 0; i < M; i++) {
//        double err = serialEllPackResult[i] - u2Hresult[i];
//        if (err < 0) err = -err;
//        if(err != 0) printf("serial = %f \t u2H = %f \t err = %f \n", serialEllPackResult[i], u2Hresult[i], err);
//        diff1 += err;
//    }
//    printf("Error: %f \n", diff1);
//
//
//    tmlt1 = 1e100;
//    double *ompEllpackH = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        double t1 = wtime();
//        ompEllpackH = ellpack.openMPMultiplyUnroll2Hor(x, y);
//        double t2 = wtime();
//        tmlt1 = dmin(tmlt1, (t2 - t1));
//    }
//    gflops1 = (2.0 * nz / tmlt1 * 1e-6) * 0.001;
//    printf("2UHR ELLPACK %d x %d: time %lf  GFLOPS: %f \n", M, N, tmlt1, gflops1);
//    //validate result
//     diff1 = 0;
//    for (int i = 0; i < M; i++) {
//        double err = ompELLPackresult[i] - ompEllpackH[i];
//        if (err < 0) err = -err;
//        diff1 += err;
//    }
//    printf("Error: %f \n", diff1);
//    tmlt1 = 1e100;
//    double *unroll2HVresult = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        double t1 = wtime();
//        unroll2HVresult = ellpack.openMPMultiplyUnroll2HorVert(x, y);
//        double t2 = wtime();
//        tmlt1 = dmin(tmlt1, (t2 - t1));
//    }
//    gflops1 = (2.0 * nz / tmlt1 * 1e-6) * 0.001;
//    printf("UHV time %lf  GFLOPS: %f \n", M, N, tmlt1, gflops1);

//    delete[] x;
//    delete[] y;
    delete[] I;
    delete[] J;
    delete[] val;

    return 0;
}