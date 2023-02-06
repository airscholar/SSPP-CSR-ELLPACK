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
    double *AS;
    unordered_map<pair<int, int>, double, hash_pair> matrixElements;

public:
    //constructor
    MatrixCSR(int rows, int cols, int nz, int *I, int *J, double *val);

    //methods
    int getRows();

    int getCols();

    int getNZ();

    int *getIRP();

    int *getJA();

    double *getAS();

    void sortData();

    void setJA(int nz, int *J);

    void setAS(int nz, double *val);

    void setIRP(int nz, int *I);

    double *serialMultiply(int *IRP);

    double *openMPMultiply(int *IRP);
};


#endif //SMALLSCALE_MATRIXCSR_H
