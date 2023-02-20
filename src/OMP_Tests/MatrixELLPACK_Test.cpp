#include "MatrixELLPACK_Test.h"

void MatrixELLPACK_Test::createTestI() {
    I[0] = 1;
    I[1] = 1;
    I[2] = 2;
    I[3] = 2;
    I[4] = 3;
    I[5] = 4;
    I[6] = 4;
}

void MatrixELLPACK_Test::createTestJ() {
    J[0] = 1;
    J[1] = 2;
    J[2] = 2;
    J[3] = 3;
    J[4] = 3;
    J[5] = 3;
    J[6] = 4;
}

void MatrixELLPACK_Test::createTestVal() {
    val[0] = 11;
    val[1] = 12;
    val[2] = 22;
    val[3] = 23;
    val[4] = 33;
    val[5] = 43;
    val[6] = 44;
}

int MatrixELLPACK_Test::isEqualDouble(int size, double * matrix1, double * matrix2) {
    for (int i = 0;i < size;i++) {
        // If the matrices are different we return 0
        if (matrix1[i] != matrix2[i])
            return 0;
    }
    return 1;
}

int MatrixELLPACK_Test::isEqualInt(int size, int * matrix1, int * matrix2) {
    for (int i = 0;i < size;i++) {
        // If the matrices are different we return 0
        if (matrix1[i] != matrix2[i])
            return 0;
    }
    return 1;
}

int* MatrixELLPACK_Test::getJA_Test() {
    return JA;
}

double* MatrixELLPACK_Test::getAS_Test() {
    return AS;
}

void MatrixELLPACK_Test::setJA_Test() {
    //ELLPACK JA
    // I and J are already sorted
    int k, p, q;
    k = 1;
    int idx;

    for (p = 1; p <= rows; p++) {
        for (q = 1; q <= maxNZ; q++) {
            idx = (p - 1) * maxNZ + (q - 1);
            if (I[k - 1] + 1 == p) {
                JA[idx] = J[k - 1];
                k++;
            } else
                JA[idx] = -1;
        }
    }
}

void MatrixELLPACK_Test::setAS_Test() {
    //call the base function
    for (int i = 0; i < nz; i++) {
        this->AS[i] = val[i];
    }
}

void MatrixELLPACK_Test::getMaxNZ_Test(int nz, int *I) {
    // We create an array that will contain the number of non-zero for each row
    // from this array we will get the max, that is MAXNZ
    int *temp = new int[nz];
    // We initialise its values to zero
    for (int i = 0; i < nz; i++) {
        temp[i] = 0;
    }

    for (int i = 0; i < nz; i++) {
        temp[I[i]]++;
    }

    int maximum = temp[0];

    for (int i = 1; i < nz; i++) {
        if (temp[i] > maximum)
            maximum = temp[i];
    }
    return maximum;
}
