//
// Created by Yusuf Ganiyu on 2/4/23.
//

#ifndef SMALLSCALE_MATRIXCSR_H
#define SMALLSCALE_MATRIXCSR_H

#include <cstdio>
#include <cstdlib>
#include "../mmio.h"
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "../MatrixBase.h"

using namespace std;

class MatrixCSR : public MatrixBase {
    int rows;
    int cols;
    int nz;
    int *I;
    int *J;
    double *val;
    int *IRP{};
    int *JA{};
    double *x;
    double *AS{};

public:
    MatrixCSR(int rows, int cols, int nz, int *I, int *J, double *val, double *x) {
        this->rows = rows;
        this->cols = cols;
        this->nz = nz;
        this->I = I;
        this->J = J;
        this->val = val;
        this->x = x;

        this->IRP = new int[rows + 1];
        this->JA = new int[nz];
        this->AS = new double[nz];

        this->setIRP(nz, I);
        this->setJA(nz, I, J);
        this->setAS(nz, val);
    }

    double *serialMultiply(double *x, double *y) override;

    double *openMPMultiplyUnroll2H(double *x, double *y) override;

    double *openMPMultiplyUnroll2V(double *x, double *y) override;

    double *openMPMultiplyUnroll4H(double *x, double *y) override;

    double *openMPMultiplyUnroll4V(double *x, double *y) override;

    double *openMPMultiplyUnroll8H(double *x, double *y) override;

    double *openMPMultiplyUnroll8V(double *x, double *y) override;

    double *openMPMultiplyUnroll16H(double *x, double *y) override;

    double *openMPMultiplyUnroll16V(double *x, double *y) override;

    int *getIRP();

    void setIRP(int nz, int *I);

    void setJA(int nz, int *I, int *J) override;

    void setAS(int nz, double *val) override;

    int *getJA() override;

    double *getAS() override;
};

#endif //SMALLSCALE_MATRIXCSR_H
