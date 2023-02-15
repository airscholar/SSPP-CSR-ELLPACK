//
// Created by Yusuf Ganiyu on 2/5/23.
//

#ifndef SMALLSCALE_MATRIXELLPACK_H
#define SMALLSCALE_MATRIXELLPACK_H
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "../MatrixBase.h"

class MatrixELLPACK: public MatrixBase{
private:
    int rows;
    int cols;
    int nz;
    int maxNZ;
    int *I;
    int *J;
    double *val;
    int *JA{};
    double *x;
    double *AS{};

public:
    MatrixELLPACK(int rows, int cols, int nz, int* &I, int* &J, double* &val, double *x) {
        this->rows = rows;
        this->cols = cols;
        this->nz = nz;
        this->I = I;
        this->J = J;
        this->val = val;
        this->x = x;

        this->maxNZ = getMaxNZ(nz, I);
        this->JA = new int[maxNZ * this->rows];
        this->AS = new double[maxNZ * this->rows];

        this->setJA(nz, I, J);
        this->setAS(maxNZ, val);

//        //print JA
//        printf("JA: ");
//        for(int i = 0; i < maxNZ * this->rows; i++){
//            printf("%d ", JA[i]);
//        }
//        printf("\n");
//
//        //print AS
//        printf("AS: ");
//        for(int i = 0; i < nz; i++){
//            printf("%.2f ", AS[i]);
//        }
//        printf("\n");
    }

    int getMaxNZ(int nz, int *I);

    void setJA(int nz, int *I, int *J);

    void setAS(int nz, double *val);

    double *serialMultiply(double *x, double *y) override;

    double *serialOMPMultiply(double *x, double *y);

    double *openMPMultiplyUnroll2H(double *x, double *y) override;

    double *openMPMultiplyUnroll2V(double *x, double *y) override;

    double *openMPMultiplyUnroll4H(double *x, double *y) override;

    double *openMPMultiplyUnroll4V(double *x, double *y) override;

    double *openMPMultiplyUnroll8H(double *x, double *y) override;

    double *openMPMultiplyUnroll8V(double *x, double *y) override;

    double *openMPMultiplyUnroll16H(double *x, double *y) override;

    double *openMPMultiplyUnroll16V(double *x, double *y) override;
};


#endif //SMALLSCALE_MATRIXELLPACK_H
