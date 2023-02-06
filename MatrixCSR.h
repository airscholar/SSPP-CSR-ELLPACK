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
    vector<int> IRP;
    vector<int> JA;
    vector<double> AS;
    unordered_map<pair<int, int>, double, hash_pair> matrixElements;

public:
    //constructor
    MatrixCSR(int rows, int cols, int nz, int *I, int *J, double *val);

    //methods
    int getRows();

    int getCols();

    int getNZ();

    vector<int> getIRP();

    vector<int> getJA();

    vector<double> getAS();

    vector<int> serialMultiply(const vector<int> v);

    vector<int> openMPMultiply(const vector<int> v);
};


#endif //SMALLSCALE_MATRIXCSR_H
