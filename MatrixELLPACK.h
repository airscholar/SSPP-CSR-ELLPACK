//
// Created by Yusuf Ganiyu on 2/5/23.
//

#ifndef SMALLSCALE_MATRIXELLPACK_H
#define SMALLSCALE_MATRIXELLPACK_H
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

class MatrixELLPACK {
private:
    int rows;
    int cols;
    int nz;
    int *I;
    int *J;
    double *val;
    int maxNZ;

    int *IRP;
    int *JA;
    double *AS;
public:
    MatrixELLPACK(int rows, int cols, int nz, int *I, int *J, double *val);

    int *getI();

    int *getJ();

    double *getVal();

    int getRows();

    int getCols();

    int getNZ();

    int getMaxNZ(int nz, int *I);

    void sortData();

    int *getJA();

    void setJA(int nz, int *I, int *J);

    double *getAS();

    void setAS(double *val);

    double *multiplyELLPack(double *x);
};


#endif //SMALLSCALE_MATRIXELLPACK_H
