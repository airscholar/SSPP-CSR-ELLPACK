#include "MatrixCSR_Test.h"

void MatrixCSR_Test::createTestI() {
    I[0] = 1;
    I[1] = 1;
    I[2] = 2;
    I[3] = 2;
    I[4] = 3;
    I[5] = 4;
    I[6] = 4;
}

void MatrixCSR_Test::createTestJ() {
    J[0] = 1;
    J[1] = 2;
    J[2] = 2;
    J[3] = 3;
    J[4] = 3;
    J[5] = 3;
    J[6] = 4;
}

void MatrixCSR_Test::createTestVal() {
    val[0] = 11;
    val[1] = 12;
    val[2] = 22;
    val[3] = 23;
    val[4] = 33;
    val[5] = 43;
    val[6] = 44;
}

int MatrixCSR_Test::isEqualDouble(int size, double * matrix1, double * matrix2) {
    for (int i = 0;i < size;i++) {
        // If the matrices are different we return 0
        if (matrix1[i] != matrix2[i])
            return 0;
    }
    return 1;
}

int MatrixCSR_Test::isEqualInt(int size, int * matrix1, int * matrix2) {
    for (int i = 0;i < size;i++) {
        // If the matrices are different we return 0
        if (matrix1[i] != matrix2[i])
            return 0;
    }
    return 1;
}

void MatrixCSR_Test::setIRP_Test() {
    IRP[0] = 0;
    IRP[1] = 2;
    IRP[2] = 4;
    IRP[3] = 5;
    IRP[4] = 7;
}

int* MatrixCSR_Test::getIRP_Test() {
    return IRP;
}

int* MatrixCSR_Test::getJA_Test() {
    return JA;
}

double* MatrixCSR_Test::getAS_Test() {
    return AS;
}

void MatrixCSR_Test::setJA_Test() {
    JA[0] = 0;
    JA[1] = 1;
    JA[2] = 1;
    JA[3] = 2;
    JA[4] = 2;
    JA[5] = 2;
    JA[6] = 3;
}

void MatrixCSR_Test::setAS_Test() {
    //call the base function
    for (int i = 0; i < nz; i++) {
        this->AS[i] = val[i];
    }
}
