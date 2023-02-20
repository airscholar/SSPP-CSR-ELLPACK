// This file contains the test functions

#ifndef OMP_TEST_H_
#define OMP_TEST_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "../MatrixBase.h"
#include "../OMP/MatrixELLPACK.h"

class MatrixELLPACK_Test {
private:
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
    int maxNZ;

public:
    MatrixELLPACK_Test(int rows, int cols, int nz, int *I, int *J, double *val) {
        this->rows = 4;
        this->cols = 4;
        this->nz = 7;
        this->I = new int[nz];
        this->J = new int[nz];
        this->val = new double[nz];

        this->x = MatrixBase::generateVector(nz);

        createTestI();
        createTestJ();
        createTestVal();

        this->maxNZ = getMaxNZ_Test(nz, I);
        this->JA = new int[maxNZ * this->rows];
        this->AS = new double[maxNZ * this->rows];

        this->setJA_Test();
        this->setAS_Test();

        //MAIN APP CODE TO TEST
        char filename[] = "../input/test.mtx";
        MatrixBase::readFile(rows, cols, nz, I, J, val, filename);
        MatrixBase::sortData(I, J, val, nz);

        MatrixELLPACK matrixELLPACK(rows, cols, nz, I, J, val, x);

//        printf("TEST");
//
//        printf("JA: ");
//        for (int i = 0; i < rows; i++) {
//            for (int j = 0; j < maxNZ; j++) {
//                cout << JA[i * maxNZ + j] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//
//        printf("AS: ");
//        for (int i = 0; i < nz; i++) {
//            cout << AS[i] << " ";
//        }
//        cout << endl;

        assert(MatrixELLPACK_Test::isEqualInt(nz, matrixELLPACK.getJA(), getJA_Test()));
        assert(MatrixELLPACK_Test::isEqualDouble(nz, matrixELLPACK.getAS(), getAS_Test()));
    }

    void createTestI();

    void createTestJ();

    void createTestVal();

    void setJA_Test();

    void setAS_Test();

    void setMaxNZ_Test();

    static int isEqualInt(int size, int *a, int *b);

    static int isEqualDouble(int size, double *a, double *b);

    int* getJA_Test();

    double* getAS_Test();

    int getMaxNZ_Test(int nz, int *I);
};

#endif