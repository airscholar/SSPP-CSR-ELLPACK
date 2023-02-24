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
public:
    MatrixELLPACK(int rows_, int cols_, int nz_, int* &I_, int* &J_, double* &val_, double *x_) {
        rows = rows_;
        cols = cols_;
        nz = nz_;
        I = I_;
        J = J_;
        val = val_;
        x = x_;

        setMaxNZ(nz, I);

        JA = new int[maxNZ * rows];
        AS = new double[maxNZ * rows];

        setJA(nz, I, J);
        setAS(maxNZ, val);

//        //print JA in matrix format
//        printf("JA: \n");
//        for(int i = 0; i < rows; i++){
//            for(int j = 0; j < maxNZ; j++){
//                printf("%d ", JA[i * maxNZ + j]);
//            }
//            printf("\n");
//        }
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
//        int *JA_T = new int[maxNZ * rows];
//        for(int i = 0; i < rows; i++){
//            for(int j = 0; j < maxNZ; j++){
//                JA_T[j * rows + i] = JA[i * maxNZ + j];
//            }
//        }
//
//        // transpose AS
//        double *AS_T = new double[maxNZ * rows];
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
