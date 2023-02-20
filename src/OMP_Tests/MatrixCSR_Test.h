// This file contains the test functions

#ifndef OMP_TEST_H_
#define OMP_TEST_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "../MatrixBase.h"
#include "../OMP/MatrixCSR.h"

class MatrixCSR_Test {
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

public:
    MatrixCSR_Test(int rows, int cols, int nz, int *I, int *J, double *val) {
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

        this->IRP = new int[rows + 1];
        this->JA = new int[nz];
        this->AS = new double[nz];

        this->setIRP_Test();
        this->setJA_Test();
        this->setAS_Test();

        //MAIN APP CODE TO TEST
        char filename[] = "../input/test.mtx";
        MatrixBase::readFile(rows, cols, nz, I, J, val, filename);
        MatrixBase::sortData(I, J, val, nz);

        MatrixCSR matrixCSR(rows, cols, nz, I, J, val, x);

//        printf("TEST");
//        printf("IRP: ");
//        for (int i = 0; i < rows + 1; i++) {
//            cout << IRP[i] << " ";
//        }
//        cout << endl;
//
//        printf("JA: ");
//        for (int i = 0; i < nz; i++) {
//            cout << JA[i] << " ";
//        }
//        cout << endl;
//
//        printf("AS: ");
//        for (int i = 0; i < nz; i++) {
//            cout << AS[i] << " ";
//        }
//        cout << endl;

        assert(MatrixCSR_Test::isEqualInt(rows + 1, matrixCSR.getIRP(), getIRP_Test()));
        assert(MatrixCSR_Test::isEqualInt(nz, matrixCSR.getJA(), getJA_Test()));
        assert(MatrixCSR_Test::isEqualDouble(nz, matrixCSR.getAS(), getAS_Test()));
    }

    void createTestI();

    void createTestJ();

    void createTestVal();

    void setIRP_Test();

    void setJA_Test();

    void setAS_Test();

    static int isEqualInt(int size, int *a, int *b);

    static int isEqualDouble(int size, double *a, double *b);

    int* getIRP_Test();

    int* getJA_Test();

    double* getAS_Test();
};

#endif