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

public:
    MatrixCSR(int rows_, int cols_, int nz_, int* &I_, int* &J_, double* &val_, double *x_) {
        rows = rows_;
        cols = cols_;
        nz = nz_;
        I = I_;
        J = J_;
        val = val_;
        x = x_;

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
