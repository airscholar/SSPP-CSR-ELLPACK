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
    double *x;
    double *val;
    int maxNZ;

    int *IRP;
    int *JA;
    double *AS;
public:
    MatrixELLPACK(int rows, int cols, int nz, int *I, int *J, double *val, double *x);

    int *getI();

    int *getJ();

    double *getVal();

    int getRows();

    int getCols();

    int getNZ();

    int getMaxNZ(int nz, int *I);

    void swap(int &a, int &b);

    void swap(double &a, double &b);

    int partition(int I[], int J[], double val[], int low, int high);

    void sortData(int I[], int J[], double val[], int low, int high);

    int *getJA();

    void setJA(int nz, int *I, int *J);

    double *getAS();

    void setAS(double *val);

    double *multiplyELLPack(double *x, double *y);

    double *OMPMultiplyELLPack(double *x, double *y);

};


#endif //SMALLSCALE_MATRIXELLPACK_H
