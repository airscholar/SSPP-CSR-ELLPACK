//
// Created by Yusuf Ganiyu on 2/4/23.
//

#ifndef SMALLSCALE_MATRIXCSR_H
#define SMALLSCALE_MATRIXCSR_H

#include <cstdio>
#include <cstdlib>
#include "mmio.h"
#include "smallscale.h"
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

using namespace std;

class MatrixCSR {

private:
    int rows;
    int cols;
    int nz;
    int *I;
    int *J;
    double *val;
    int *IRP;
    int *JA;
    double* x;
    double *AS;

public:
    //constructor
    MatrixCSR(int rows, int cols, int nz, int *I, int *J, double *val, double* x);

    //methods
    int getRows();

    int getCols();

    int getNZ();

    int *getIRP();

    int *getJA();

    double *getAS();

    void swap(int &a, int &b);

    void swap(double &a, double &b);

    int partition(int I[], int J[], double val[], int low, int high);

    void sortData(int I[], int J[], double val[], int low, int high);

    void setJA(int nz, int *J);

    void setAS(int nz, double *val);

    void setIRP(int nz, int *I);

    double* generateVector(int rows,  int cols);

    double *serialMultiply(double *v);

    double *openMPMultiply(double *v);
};


#endif //SMALLSCALE_MATRIXCSR_H
