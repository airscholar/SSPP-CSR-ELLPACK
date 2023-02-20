//
// Created by Yusuf Ganiyu on 2/14/23.
//

#include <utility>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <utility>
#include <algorithm>
#include <map>

#ifndef SMALLSCALE_MATRIXBASE_H
#define SMALLSCALE_MATRIXBASE_H

using namespace std;

class MatrixBase {
private:
    map<pair<int, int>, double> matrix; // use a map to store the values of I, J and V

public:
    static double *generateVector(int size);

    static void readFile(int &M, int &N, int &nz, int *&I, int *&J, double *&val, char *fileName);

    static void sortData(int *&I, int *&J, double *&AS, int nz);

    static double compute_Max_Error(double *x, double *y, int rows);

    static void CSR_CpuMatrixVector(int rows, int *IRP, int *JA, double *AS, double *x, double *y);

    static void ELL_CpuMatrixVector(int rows, int *JA, double *AS, int maxNZ, double *x, double *y);

    void printMatrix(int M, int N, bool showImage);

    virtual double *getAS() = 0;

    virtual int *getJA() = 0;

    virtual void setJA(int nz, int *I, int *J) = 0;

    virtual void setAS(int maxNZ, double *val) = 0;

    virtual double *serialMultiply(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll2H(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll2V(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll4H(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll4V(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll8H(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll8V(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll16H(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll16V(double *x, double *y) = 0;
};


#endif //SMALLSCALE_MATRIXBASE_H
