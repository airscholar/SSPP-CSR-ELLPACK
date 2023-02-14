//
// Created by Yusuf Ganiyu on 2/5/23.
//

#ifndef SMALLSCALE_MATRIXELLPACK_H
#define SMALLSCALE_MATRIXELLPACK_H
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "MatrixBase.h"

class MatrixELLPACK: public MatrixBase{
private:
    int rows;
    int cols;
    int nz;
    int maxNZ;
    int *I;
    int *J;
    double *val;
    int *IRP{};
    //declare JA as a multidimensioanl array
    std::vector<std::vector<int>> JA;
    double *x;
    double *AS{};

public:
    MatrixELLPACK(int rows, int cols, int nz, int *I, int *J, double *val, double *x): MatrixBase(rows, cols, nz,  I, J, val, x) {
        this->rows = rows;
        this->cols = cols;
        this->nz = nz;
        this->I = I;
        this->J = J;
        this->val = val;
        this->x = x;

        this->maxNZ = getMaxNZ(nz, I);
        this->JA = std::vector<std::vector<int>>(rows, std::vector<int>(maxNZ, 0));
        this->AS = new double[nz];

        this->setJA(nz, I, J);
        this->setAS(maxNZ, val);

        this->sortData(I, J, val, nz);

    }

    int getMaxNZ(int nz, int *I);

    double *getAS();

    void setJA(int nz, int *I, int *J);

    std::vector<std::vector<int>> getJA();

    void setAS(int nz, double *val);

    double *serialMultiply(double *x, double *y) override;

    double *openMPMultiply(double *x, double *y) override;

};


#endif //SMALLSCALE_MATRIXELLPACK_H
