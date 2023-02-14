#include <vector>
#include "MatrixCSR.h"

double *MatrixCSR::serialMultiply(double *v, double *y) {
#pragma omp parallel for shared(y, v)
    for (int i = 0; i < this->rows; i++) {
        double t = 0;
        for (int j = this->getIRP()[i]; j < this->getIRP()[i + 1]; j++) {
            t += this->getAS()[j] * v[this->getJA()[j]];
        }
        y[i] = t;
    }

    return y;
}

double *MatrixCSR::openMPMultiply(double *v, double *y) {
    double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    int i, j, k;
    int tile_size = 64; // adjust this value to a suitable value for your system
#pragma omp parallel for private(i, j, k, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15) shared(y, v)
    for (i = 0; i < rows; i += tile_size) {
        for (k = i; k < std::min(rows, i + tile_size); k++) {
            t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0, t14 = 0, t15 = 0;
            for (j = this->getIRP()[k]; j < this->getIRP()[k + 1] - 15; j += 16) {
                t0 += this->getAS()[j] * v[this->getJA()[j]];
                t1 += this->getAS()[j + 1] * v[this->getJA()[j + 1]];
                t2 += this->getAS()[j + 2] * v[this->getJA()[j + 2]];
                t3 += this->getAS()[j + 3] * v[this->getJA()[j + 3]];
                t4 += this->getAS()[j + 4] * v[this->getJA()[j + 4]];
                t5 += this->getAS()[j + 5] * v[this->getJA()[j + 5]];
                t6 += this->getAS()[j + 6] * v[this->getJA()[j + 6]];
                t7 += this->getAS()[j + 7] * v[this->getJA()[j + 7]];
                t8 += this->getAS()[j + 8] * v[this->getJA()[j + 8]];
                t9 += this->getAS()[j + 9] * v[this->getJA()[j + 9]];
                t10 += this->getAS()[j + 10] * v[this->getJA()[j + 10]];
                t11 += this->getAS()[j + 11] * v[this->getJA()[j + 11]];
                t12 += this->getAS()[j + 12] * v[this->getJA()[j + 12]];
                t13 += this->getAS()[j + 13] * v[this->getJA()[j + 13]];
                t14 += this->getAS()[j + 14] * v[this->getJA()[j + 14]];
                t15 += this->getAS()[j + 15] * v[this->getJA()[j + 15]];
            }
            for (; j < this->getIRP()[k + 1]; j++) {
                t0 += this->getAS()[j] * v[this->getJA()[j]];
            }
            y[k] = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t14 + t15;
//        y[i] = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t14 + t15;
        }
    }
    return y;
}

int* MatrixCSR::getIRP() {
    return this->IRP;
}

int* MatrixCSR::getJA() {
    return this->JA;
}

double* MatrixCSR::getAS() {
    return this->AS;
}

void MatrixCSR::setIRP(int nz, int *I) {
    // Step 1: We count the number of non-zero elements per row
    int *count = new int[this->rows];
    for (int i = 0; i < this->rows; i++) {
        count[i] = 0;
    }
    for (int i = 0; i < nz; i++) {
        count[I[i]]++;
    }

    // Step 2: We compute the IRP array
    this->IRP[0] = 0;
    for (int i = 1; i < this->rows + 1; i++) {
        this->IRP[i] = this->IRP[i - 1] + count[i - 1];
    }
}

void MatrixCSR::setJA(int nz, int *I, int *J) {
    //call the base function
    for (int i = 0; i < nz; i++) {
        this->JA[i] = J[i];
    }
}

void MatrixCSR::setAS(int nz, double *val) {
    //call the base function
    for (int i = 0; i < nz; i++) {
        this->AS[i] = val[i];
    }
}
