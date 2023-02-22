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

        this->setMaxNZ(nz, I);
        this->JA = new int[maxNZ * this->rows];
        this->AS = new double[maxNZ * this->rows];

        this->setJA(nz, I, J);
        this->setAS(maxNZ, val);

//        //print JA in matrix format
//        printf("JA: \n");
//        for(int i = 0; i < rows; i++){
//            for(int j = 0; j < maxNZ; j++){
//                printf("%d ", JA[i * maxNZ + j]);
//            }
//            printf("\n");
//        }
//
//        //print AS
//        printf("AS: ");
//        for(int i = 0; i < rows; i++){
//            for (int j = 0; j < maxNZ; j++){
//                printf("%f ", AS[i * maxNZ + j]);
//            }
//            printf("\n");
//        }
//        printf("\n");
//
//        // transpose JA
//        int *JA_T = new int[maxNZ * this->rows];
//        for(int i = 0; i < rows; i++){
//            for(int j = 0; j < maxNZ; j++){
//                JA_T[j * rows + i] = JA[i * maxNZ + j];
//            }
//        }
//
//        // transpose AS
//        double *AS_T = new double[maxNZ * this->rows];
//        for(int i = 0; i < rows; i++){
//            for(int j = 0; j < maxNZ; j++){
//                AS_T[j * rows + i] = AS[i * maxNZ + j];
//            }
//        }
//
//        // print JA_T
//        printf("JA_T: \n");
//        for(int i = 0; i < maxNZ; i++){
//            for(int j = 0; j < rows; j++){
//                printf("%d ", JA_T[i * rows + j]);
//            }
//            printf("\n");
//        }
//
//        // print AS_T
//        printf("AS_T: \n");
//        for(int i = 0; i < maxNZ; i++){
//            for(int j = 0; j < rows; j++){
//                printf("%f ", AS_T[i * rows + j]);
//            }
//            printf("\n");
//        }

    }

    void setMaxNZ(int nz, int *I);

    int getMaxNZ();

    void setJA(int nz, int *I, int *J) override;

    void setAS(int nz, double *val) override;

    double* getAS() override;

    int* getJA() override;

    double *serialMultiply(double *x, double *y) override;

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
