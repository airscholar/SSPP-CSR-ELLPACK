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

    std::vector<std::vector<int>> JA;
    std::vector<std::vector<double>> AS;
public:
    MatrixELLPACK(int rows, int cols, int nz, int *I, int *J, double *val);

    int* getI();

    int* getJ();

    double* getVal();

    int getRows();

    int getCols();

    int getNZ();

    int getMaxNZ();

    std::vector<std::vector<int>> getJA();

    std::vector<std::vector<double>> getAS();

    std::vector<int> multiply(int rows, int maxNZ, std::vector<std::vector<int>> JA, std::vector<std::vector<double>> AS, std::vector<double> x);
};


#endif //SMALLSCALE_MATRIXELLPACK_H
