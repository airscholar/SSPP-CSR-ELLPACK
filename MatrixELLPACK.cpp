//
// Created by Yusuf Ganiyu on 2/5/23.
//
//#include "smallscale.h"
#include <vector>
#include "MatrixELLPACK.h"
#include <omp.h>

MatrixELLPACK::MatrixELLPACK(int rows, int cols, int nz, int *I, int *J, double *val, double* x) {
    this->rows = rows;
    this->cols = cols;
    this->nz = nz;
    this->I = I;
    this->J = J;
    this->val = val;
    this->x = x;
    this->maxNZ = getMaxNZ(nz, I);

//    printf("Starting conversion to ELLPACK\n");
    //sort the data I, J, val
//    sortData(I, J, val, 0, nz - 1);
//    printf("Data sorted\n");

    setJA(nz, I, J);
//    printf("JA set\n");

    setAS(val);
//    printf("AS set\n");
}

void MatrixELLPACK::swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

void MatrixELLPACK::swap(double &a, double &b) {
    double temp = a;
    a = b;
    b = temp;
}

int MatrixELLPACK::partition(int I[], int J[], double val[], int low, int high) {
    int pivotValue = I[high];
    int i = low - 1;
    for (int j = low; j <= high - 1; ++j) {
        if (I[j] < pivotValue) {
            ++i;
            swap(I[i], I[j]);
            swap(J[i], J[j]);
            swap(val[i], val[j]);
        }
    }
    swap(I[i + 1], I[high]);
    swap(J[i + 1], J[high]);
    swap(val[i + 1], val[high]);
    return i + 1;
}

void MatrixELLPACK::sortData(int I[], int J[], double val[], int low, int high) {
//    if (low < high) {
//        int pivotIndex = partition(I, J, val, low, high);
//        sortData(I, J, val, low, pivotIndex - 1);
//        sortData(I, J, val, pivotIndex + 1, high);
//    }
    int i, j;
    int temp;
    double tempVal;
    for (i = 0; i < nz - 1; ++i) {
        for (j = 0; j < nz - 1 - i; ++j) {
            if (I[j] > I[j + 1]) {
                temp = I[j + 1];
                I[j + 1] = I[j];
                I[j] = temp;
                temp = J[j + 1];
                J[j + 1] = J[j];
                J[j] = temp;
                tempVal = val[j + 1];
                val[j + 1] = val[j];
                val[j] = tempVal;
            }
        }
    }
}

//void MatrixELLPACK::sortData(){
//    int i, j;
//    int temp;
//    double tempVal;
//    for (i = 0; i < nz - 1; ++i) {
//        for (j = 0; j < nz - 1 - i; ++j) {
//            if (I[j] > I[j + 1]) {
//                temp = I[j + 1];
//                I[j + 1] = I[j];
//                I[j] = temp;
//                temp = J[j + 1];
//                J[j + 1] = J[j];
//                J[j] = temp;
//                tempVal = val[j + 1];
//                val[j + 1] = val[j];
//                val[j] = tempVal;
//            }
//        }
//    }
//}

// Returns the MAXNZ of the ELLPACK
int MatrixELLPACK::getMaxNZ(int nz, int *I) {
    // We create an array that will contain the number of non zero for each row
    // from this array we will get the max, that is MAXNZ
    int *temp = (int *) malloc(nz * sizeof(int));
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

int* MatrixELLPACK::getJA() {
    return this->JA;
}

void MatrixELLPACK::setJA(int nz, int *I, int *J) {
    this->JA = new int[rows * maxNZ];
    // Returns the JA of the ELLPACK
    // Here we use the reordered I and J
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
                JA[idx] = J[k - 2];
        }
    }
}

double* MatrixELLPACK::getAS() {
    return this->AS;
}

void MatrixELLPACK::setAS(double *val) {
    this->AS = new double[rows * maxNZ];
    // Returns the AS of the ELLPACK
    int x, y, z;
    int l = 1;
    int idx;

    for (x = 1; x <= rows; x++) {
        for (y = 1; y <= maxNZ; y++) {
            idx = (x - 1) * maxNZ + (y - 1);
            if (I[l - 1] + 1 == x) {
                AS[idx] = val[l - 1];
                l++;
            }
//            else
//                AS[idx] = val[k - 2];
        }
    }
}

double* MatrixELLPACK::multiplyELLPack(double* x, double *y) {
    double t;
    int i, j, idx; // idx is the index of (i,j)
#pragma omp parallel for shared(x, y) private(t, i, j, idx)
    for (i = 0; i < rows; i++) {
        t = 0;
        for (j = 0; j < maxNZ; j++) {
            idx = i * maxNZ + j;
            t += AS[idx] * x[JA[idx]];
        }
        y[i] = t;
    }
    return y;
}

double* MatrixELLPACK::OMPMultiplyELLPack(double* x, double *y) {
    double t, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    int i, j, idx; // idx is the index of (i,j)

#pragma omp parallel for shared(x, y) private(t, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, i, j, idx)
    for (i = 0; i < cols - cols % 16; i += 16) {
        t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0, t14 = 0, t15 = 0;

        for (j = 0; j < maxNZ - maxNZ % 16; j += 16) {
            t0 += AS[(i + 0) * maxNZ + j + 0] * x[JA[(i + 0) * maxNZ + j + 0]] +
                  AS[(i + 0) * maxNZ + j + 1] * x[JA[(i + 0) * maxNZ + j + 1]] +
                  AS[(i + 0) * maxNZ + j + 2] * x[JA[(i + 0) * maxNZ + j + 2]] +
                  AS[(i + 0) * maxNZ + j + 3] * x[JA[(i + 0) * maxNZ + j + 3]] +
                  AS[(i + 0) * maxNZ + j + 4] * x[JA[(i + 0) * maxNZ + j + 4]] +
                  AS[(i + 0) * maxNZ + j + 5] * x[JA[(i + 0) * maxNZ + j + 5]] +
                  AS[(i + 0) * maxNZ + j + 6] * x[JA[(i + 0) * maxNZ + j + 6]] +
                  AS[(i + 0) * maxNZ + j + 7] * x[JA[(i + 0) * maxNZ + j + 7]] +
                  AS[(i + 0) * maxNZ + j + 8] * x[JA[(i + 0) * maxNZ + j + 8]] +
                  AS[(i + 0) * maxNZ + j + 9] * x[JA[(i + 0) * maxNZ + j + 9]] +
                  AS[(i + 0) * maxNZ + j + 10] * x[JA[(i + 0) * maxNZ + j + 10]] +
                  AS[(i + 0) * maxNZ + j + 11] * x[JA[(i + 0) * maxNZ + j + 11]] +
                  AS[(i + 0) * maxNZ + j + 12] * x[JA[(i + 0) * maxNZ + j + 12]] +
                  AS[(i + 0) * maxNZ + j + 13] * x[JA[(i + 0) * maxNZ + j + 13]] +
                  AS[(i + 0) * maxNZ + j + 14] * x[JA[(i + 0) * maxNZ + j + 14]] +
                  AS[(i + 0) * maxNZ + j + 15] * x[JA[(i + 0) * maxNZ + j + 15]];
            t1 += AS[(i + 1) * maxNZ + j + 0] * x[JA[(i + 1) * maxNZ + j + 0]] +
                  AS[(i + 1) * maxNZ + j + 1] * x[JA[(i + 1) * maxNZ + j + 1]] +
                  AS[(i + 1) * maxNZ + j + 2] * x[JA[(i + 1) * maxNZ + j + 2]] +
                  AS[(i + 1) * maxNZ + j + 3] * x[JA[(i + 1) * maxNZ + j + 3]] +
                  AS[(i + 1) * maxNZ + j + 4] * x[JA[(i + 1) * maxNZ + j + 4]] +
                  AS[(i + 1) * maxNZ + j + 5] * x[JA[(i + 1) * maxNZ + j + 5]] +
                  AS[(i + 1) * maxNZ + j + 6] * x[JA[(i + 1) * maxNZ + j + 6]] +
                  AS[(i + 1) * maxNZ + j + 7] * x[JA[(i + 1) * maxNZ + j + 7]] +
                  AS[(i + 1) * maxNZ + j + 8] * x[JA[(i + 1) * maxNZ + j + 8]] +
                  AS[(i + 1) * maxNZ + j + 9] * x[JA[(i + 1) * maxNZ + j + 9]] +
                  AS[(i + 1) * maxNZ + j + 10] * x[JA[(i + 1) * maxNZ + j + 10]] +
                  AS[(i + 1) * maxNZ + j + 11] * x[JA[(i + 1) * maxNZ + j + 11]] +
                  AS[(i + 1) * maxNZ + j + 12] * x[JA[(i + 1) * maxNZ + j + 12]] +
                  AS[(i + 1) * maxNZ + j + 13] * x[JA[(i + 1) * maxNZ + j + 13]] +
                  AS[(i + 1) * maxNZ + j + 14] * x[JA[(i + 1) * maxNZ + j + 14]] +
                  AS[(i + 1) * maxNZ + j + 15] * x[JA[(i + 1) * maxNZ + j + 15]];
            t2 += AS[(i + 2) * maxNZ + j + 0] * x[JA[(i + 2) * maxNZ + j + 0]] +
                  AS[(i + 2) * maxNZ + j + 1] * x[JA[(i + 2) * maxNZ + j + 1]] +
                  AS[(i + 2) * maxNZ + j + 2] * x[JA[(i + 2) * maxNZ + j + 2]] +
                  AS[(i + 2) * maxNZ + j + 3] * x[JA[(i + 2) * maxNZ + j + 3]] +
                  AS[(i + 2) * maxNZ + j + 4] * x[JA[(i + 2) * maxNZ + j + 4]] +
                  AS[(i + 2) * maxNZ + j + 5] * x[JA[(i + 2) * maxNZ + j + 5]] +
                  AS[(i + 2) * maxNZ + j + 6] * x[JA[(i + 2) * maxNZ + j + 6]] +
                  AS[(i + 2) * maxNZ + j + 7] * x[JA[(i + 2) * maxNZ + j + 7]] +
                  AS[(i + 2) * maxNZ + j + 8] * x[JA[(i + 2) * maxNZ + j + 8]] +
                  AS[(i + 2) * maxNZ + j + 9] * x[JA[(i + 2) * maxNZ + j + 9]] +
                  AS[(i + 2) * maxNZ + j + 10] * x[JA[(i + 2) * maxNZ + j + 10]] +
                  AS[(i + 2) * maxNZ + j + 11] * x[JA[(i + 2) * maxNZ + j + 11]] +
                  AS[(i + 2) * maxNZ + j + 12] * x[JA[(i + 2) * maxNZ + j + 12]] +
                  AS[(i + 2) * maxNZ + j + 13] * x[JA[(i + 2) * maxNZ + j + 13]] +
                  AS[(i + 2) * maxNZ + j + 14] * x[JA[(i + 2) * maxNZ + j + 14]] +
                  AS[(i + 2) * maxNZ + j + 15] * x[JA[(i + 2) * maxNZ + j + 15]];
            t3 += AS[(i + 3) * maxNZ + j + 0] * x[JA[(i + 3) * maxNZ + j + 0]] +
                  AS[(i + 3) * maxNZ + j + 1] * x[JA[(i + 3) * maxNZ + j + 1]] +
                  AS[(i + 3) * maxNZ + j + 2] * x[JA[(i + 3) * maxNZ + j + 2]] +
                  AS[(i + 3) * maxNZ + j + 3] * x[JA[(i + 3) * maxNZ + j + 3]] +
                  AS[(i + 3) * maxNZ + j + 4] * x[JA[(i + 3) * maxNZ + j + 4]] +
                  AS[(i + 3) * maxNZ + j + 5] * x[JA[(i + 3) * maxNZ + j + 5]] +
                  AS[(i + 3) * maxNZ + j + 6] * x[JA[(i + 3) * maxNZ + j + 6]] +
                  AS[(i + 3) * maxNZ + j + 7] * x[JA[(i + 3) * maxNZ + j + 7]] +
                  AS[(i + 3) * maxNZ + j + 8] * x[JA[(i + 3) * maxNZ + j + 8]] +
                  AS[(i + 3) * maxNZ + j + 9] * x[JA[(i + 3) * maxNZ + j + 9]] +
                  AS[(i + 3) * maxNZ + j + 10] * x[JA[(i + 3) * maxNZ + j + 10]] +
                  AS[(i + 3) * maxNZ + j + 11] * x[JA[(i + 3) * maxNZ + j + 11]] +
                  AS[(i + 3) * maxNZ + j + 12] * x[JA[(i + 3) * maxNZ + j + 12]] +
                  AS[(i + 3) * maxNZ + j + 13] * x[JA[(i + 3) * maxNZ + j + 13]] +
                  AS[(i + 3) * maxNZ + j + 14] * x[JA[(i + 3) * maxNZ + j + 14]] +
                  AS[(i + 3) * maxNZ + j + 15] * x[JA[(i + 3) * maxNZ + j + 15]];
            t4 += AS[(i + 4) * maxNZ + j + 0] * x[JA[(i + 4) * maxNZ + j + 0]] +
                  AS[(i + 4) * maxNZ + j + 1] * x[JA[(i + 4) * maxNZ + j + 1]] +
                  AS[(i + 4) * maxNZ + j + 2] * x[JA[(i + 4) * maxNZ + j + 2]] +
                  AS[(i + 4) * maxNZ + j + 3] * x[JA[(i + 4) * maxNZ + j + 3]] +
                  AS[(i + 4) * maxNZ + j + 4] * x[JA[(i + 4) * maxNZ + j + 4]] +
                  AS[(i + 4) * maxNZ + j + 5] * x[JA[(i + 4) * maxNZ + j + 5]] +
                  AS[(i + 4) * maxNZ + j + 6] * x[JA[(i + 4) * maxNZ + j + 6]] +
                  AS[(i + 4) * maxNZ + j + 7] * x[JA[(i + 4) * maxNZ + j + 7]] +
                  AS[(i + 4) * maxNZ + j + 8] * x[JA[(i + 4) * maxNZ + j + 8]] +
                  AS[(i + 4) * maxNZ + j + 9] * x[JA[(i + 4) * maxNZ + j + 9]] +
                  AS[(i + 4) * maxNZ + j + 10] * x[JA[(i + 4) * maxNZ + j + 10]] +
                  AS[(i + 4) * maxNZ + j + 11] * x[JA[(i + 4) * maxNZ + j + 11]] +
                  AS[(i + 4) * maxNZ + j + 12] * x[JA[(i + 4) * maxNZ + j + 12]] +
                  AS[(i + 4) * maxNZ + j + 13] * x[JA[(i + 4) * maxNZ + j + 13]] +
                  AS[(i + 4) * maxNZ + j + 14] * x[JA[(i + 4) * maxNZ + j + 14]] +
                  AS[(i + 4) * maxNZ + j + 15] * x[JA[(i + 4) * maxNZ + j + 15]];
            t5 += AS[(i + 5) * maxNZ + j + 0] * x[JA[(i + 5) * maxNZ + j + 0]] +
                  AS[(i + 5) * maxNZ + j + 1] * x[JA[(i + 5) * maxNZ + j + 1]] +
                  AS[(i + 5) * maxNZ + j + 2] * x[JA[(i + 5) * maxNZ + j + 2]] +
                  AS[(i + 5) * maxNZ + j + 3] * x[JA[(i + 5) * maxNZ + j + 3]] +
                  AS[(i + 5) * maxNZ + j + 4] * x[JA[(i + 5) * maxNZ + j + 4]] +
                  AS[(i + 5) * maxNZ + j + 5] * x[JA[(i + 5) * maxNZ + j + 5]] +
                  AS[(i + 5) * maxNZ + j + 6] * x[JA[(i + 5) * maxNZ + j + 6]] +
                  AS[(i + 5) * maxNZ + j + 7] * x[JA[(i + 5) * maxNZ + j + 7]] +
                  AS[(i + 5) * maxNZ + j + 8] * x[JA[(i + 5) * maxNZ + j + 8]] +
                  AS[(i + 5) * maxNZ + j + 9] * x[JA[(i + 5) * maxNZ + j + 9]] +
                  AS[(i + 5) * maxNZ + j + 10] * x[JA[(i + 5) * maxNZ + j + 10]] +
                  AS[(i + 5) * maxNZ + j + 11] * x[JA[(i + 5) * maxNZ + j + 11]] +
                  AS[(i + 5) * maxNZ + j + 12] * x[JA[(i + 5) * maxNZ + j + 12]] +
                  AS[(i + 5) * maxNZ + j + 13] * x[JA[(i + 5) * maxNZ + j + 13]] +
                  AS[(i + 5) * maxNZ + j + 14] * x[JA[(i + 5) * maxNZ + j + 14]] +
                  AS[(i + 5) * maxNZ + j + 15] * x[JA[(i + 5) * maxNZ + j + 15]];
            t6 += AS[(i + 6) * maxNZ + j + 0] * x[JA[(i + 6) * maxNZ + j + 0]] +
                  AS[(i + 6) * maxNZ + j + 1] * x[JA[(i + 6) * maxNZ + j + 1]] +
                  AS[(i + 6) * maxNZ + j + 2] * x[JA[(i + 6) * maxNZ + j + 2]] +
                  AS[(i + 6) * maxNZ + j + 3] * x[JA[(i + 6) * maxNZ + j + 3]] +
                  AS[(i + 6) * maxNZ + j + 4] * x[JA[(i + 6) * maxNZ + j + 4]] +
                  AS[(i + 6) * maxNZ + j + 5] * x[JA[(i + 6) * maxNZ + j + 5]] +
                  AS[(i + 6) * maxNZ + j + 6] * x[JA[(i + 6) * maxNZ + j + 6]] +
                  AS[(i + 6) * maxNZ + j + 7] * x[JA[(i + 6) * maxNZ + j + 7]] +
                  AS[(i + 6) * maxNZ + j + 8] * x[JA[(i + 6) * maxNZ + j + 8]] +
                  AS[(i + 6) * maxNZ + j + 9] * x[JA[(i + 6) * maxNZ + j + 9]] +
                  AS[(i + 6) * maxNZ + j + 10] * x[JA[(i + 6) * maxNZ + j + 10]] +
                  AS[(i + 6) * maxNZ + j + 11] * x[JA[(i + 6) * maxNZ + j + 11]] +
                  AS[(i + 6) * maxNZ + j + 12] * x[JA[(i + 6) * maxNZ + j + 12]] +
                  AS[(i + 6) * maxNZ + j + 13] * x[JA[(i + 6) * maxNZ + j + 13]] +
                  AS[(i + 6) * maxNZ + j + 14] * x[JA[(i + 6) * maxNZ + j + 14]] +
                  AS[(i + 6) * maxNZ + j + 15] * x[JA[(i + 6) * maxNZ + j + 15]];
            t7 += AS[(i + 7) * maxNZ + j + 0] * x[JA[(i + 7) * maxNZ + j + 0]] +
                  AS[(i + 7) * maxNZ + j + 1] * x[JA[(i + 7) * maxNZ + j + 1]] +
                  AS[(i + 7) * maxNZ + j + 2] * x[JA[(i + 7) * maxNZ + j + 2]] +
                  AS[(i + 7) * maxNZ + j + 3] * x[JA[(i + 7) * maxNZ + j + 3]] +
                  AS[(i + 7) * maxNZ + j + 4] * x[JA[(i + 7) * maxNZ + j + 4]] +
                  AS[(i + 7) * maxNZ + j + 5] * x[JA[(i + 7) * maxNZ + j + 5]] +
                  AS[(i + 7) * maxNZ + j + 6] * x[JA[(i + 7) * maxNZ + j + 6]] +
                  AS[(i + 7) * maxNZ + j + 7] * x[JA[(i + 7) * maxNZ + j + 7]] +
                  AS[(i + 7) * maxNZ + j + 8] * x[JA[(i + 7) * maxNZ + j + 8]] +
                  AS[(i + 7) * maxNZ + j + 9] * x[JA[(i + 7) * maxNZ + j + 9]] +
                  AS[(i + 7) * maxNZ + j + 10] * x[JA[(i + 7) * maxNZ + j + 10]] +
                  AS[(i + 7) * maxNZ + j + 11] * x[JA[(i + 7) * maxNZ + j + 11]] +
                  AS[(i + 7) * maxNZ + j + 12] * x[JA[(i + 7) * maxNZ + j + 12]] +
                  AS[(i + 7) * maxNZ + j + 13] * x[JA[(i + 7) * maxNZ + j + 13]] +
                  AS[(i + 7) * maxNZ + j + 14] * x[JA[(i + 7) * maxNZ + j + 14]] +
                  AS[(i + 7) * maxNZ + j + 15] * x[JA[(i + 7) * maxNZ + j + 15]];
            t8 += AS[(i + 8) * maxNZ + j + 0] * x[JA[(i + 8) * maxNZ + j + 0]] +
                  AS[(i + 8) * maxNZ + j + 1] * x[JA[(i + 8) * maxNZ + j + 1]] +
                  AS[(i + 8) * maxNZ + j + 2] * x[JA[(i + 8) * maxNZ + j + 2]] +
                  AS[(i + 8) * maxNZ + j + 3] * x[JA[(i + 8) * maxNZ + j + 3]] +
                  AS[(i + 8) * maxNZ + j + 4] * x[JA[(i + 8) * maxNZ + j + 4]] +
                  AS[(i + 8) * maxNZ + j + 5] * x[JA[(i + 8) * maxNZ + j + 5]] +
                  AS[(i + 8) * maxNZ + j + 6] * x[JA[(i + 8) * maxNZ + j + 6]] +
                  AS[(i + 8) * maxNZ + j + 7] * x[JA[(i + 8) * maxNZ + j + 7]] +
                  AS[(i + 8) * maxNZ + j + 8] * x[JA[(i + 8) * maxNZ + j + 8]] +
                  AS[(i + 8) * maxNZ + j + 9] * x[JA[(i + 8) * maxNZ + j + 9]] +
                  AS[(i + 8) * maxNZ + j + 10] * x[JA[(i + 8) * maxNZ + j + 10]] +
                  AS[(i + 8) * maxNZ + j + 11] * x[JA[(i + 8) * maxNZ + j + 11]] +
                  AS[(i + 8) * maxNZ + j + 12] * x[JA[(i + 8) * maxNZ + j + 12]] +
                  AS[(i + 8) * maxNZ + j + 13] * x[JA[(i + 8) * maxNZ + j + 13]] +
                  AS[(i + 8) * maxNZ + j + 14] * x[JA[(i + 8) * maxNZ + j + 14]] +
                  AS[(i + 8) * maxNZ + j + 15] * x[JA[(i + 8) * maxNZ + j + 15]];
            t9 += AS[(i + 9) * maxNZ + j + 0] * x[JA[(i + 9) * maxNZ + j + 0]] +
                  AS[(i + 9) * maxNZ + j + 1] * x[JA[(i + 9) * maxNZ + j + 1]] +
                  AS[(i + 9) * maxNZ + j + 2] * x[JA[(i + 9) * maxNZ + j + 2]] +
                  AS[(i + 9) * maxNZ + j + 3] * x[JA[(i + 9) * maxNZ + j + 3]] +
                  AS[(i + 9) * maxNZ + j + 4] * x[JA[(i + 9) * maxNZ + j + 4]] +
                  AS[(i + 9) * maxNZ + j + 5] * x[JA[(i + 9) * maxNZ + j + 5]] +
                  AS[(i + 9) * maxNZ + j + 6] * x[JA[(i + 9) * maxNZ + j + 6]] +
                  AS[(i + 9) * maxNZ + j + 7] * x[JA[(i + 9) * maxNZ + j + 7]] +
                  AS[(i + 9) * maxNZ + j + 8] * x[JA[(i + 9) * maxNZ + j + 8]] +
                  AS[(i + 9) * maxNZ + j + 9] * x[JA[(i + 9) * maxNZ + j + 9]] +
                  AS[(i + 9) * maxNZ + j + 10] * x[JA[(i + 9) * maxNZ + j + 10]] +
                  AS[(i + 9) * maxNZ + j + 11] * x[JA[(i + 9) * maxNZ + j + 11]] +
                  AS[(i + 9) * maxNZ + j + 12] * x[JA[(i + 9) * maxNZ + j + 12]] +
                  AS[(i + 9) * maxNZ + j + 13] * x[JA[(i + 9) * maxNZ + j + 13]] +
                  AS[(i + 9) * maxNZ + j + 14] * x[JA[(i + 9) * maxNZ + j + 14]] +
                  AS[(i + 9) * maxNZ + j + 15] * x[JA[(i + 9) * maxNZ + j + 15]];
            t10 += AS[(i + 10) * maxNZ + j + 0] * x[JA[(i + 10) * maxNZ + j + 0]] +
                   AS[(i + 10) * maxNZ + j + 1] * x[JA[(i + 10) * maxNZ + j + 1]] +
                   AS[(i + 10) * maxNZ + j + 2] * x[JA[(i + 10) * maxNZ + j + 2]] +
                   AS[(i + 10) * maxNZ + j + 3] * x[JA[(i + 10) * maxNZ + j + 3]] +
                   AS[(i + 10) * maxNZ + j + 4] * x[JA[(i + 10) * maxNZ + j + 4]] +
                   AS[(i + 10) * maxNZ + j + 5] * x[JA[(i + 10) * maxNZ + j + 5]] +
                   AS[(i + 10) * maxNZ + j + 6] * x[JA[(i + 10) * maxNZ + j + 6]] +
                   AS[(i + 10) * maxNZ + j + 7] * x[JA[(i + 10) * maxNZ + j + 7]] +
                   AS[(i + 10) * maxNZ + j + 8] * x[JA[(i + 10) * maxNZ + j + 8]] +
                   AS[(i + 10) * maxNZ + j + 9] * x[JA[(i + 10) * maxNZ + j + 9]] +
                   AS[(i + 10) * maxNZ + j + 10] * x[JA[(i + 10) * maxNZ + j + 10]] +
                   AS[(i + 10) * maxNZ + j + 11] * x[JA[(i + 10) * maxNZ + j + 11]] +
                   AS[(i + 10) * maxNZ + j + 12] * x[JA[(i + 10) * maxNZ + j + 12]] +
                   AS[(i + 10) * maxNZ + j + 13] * x[JA[(i + 10) * maxNZ + j + 13]] +
                   AS[(i + 10) * maxNZ + j + 14] * x[JA[(i + 10) * maxNZ + j + 14]] +
                   AS[(i + 10) * maxNZ + j + 15] * x[JA[(i + 10) * maxNZ + j + 15]];
            t11 += AS[(i + 11) * maxNZ + j + 0] * x[JA[(i + 11) * maxNZ + j + 0]] +
                   AS[(i + 11) * maxNZ + j + 1] * x[JA[(i + 11) * maxNZ + j + 1]] +
                   AS[(i + 11) * maxNZ + j + 2] * x[JA[(i + 11) * maxNZ + j + 2]] +
                   AS[(i + 11) * maxNZ + j + 3] * x[JA[(i + 11) * maxNZ + j + 3]] +
                   AS[(i + 11) * maxNZ + j + 4] * x[JA[(i + 11) * maxNZ + j + 4]] +
                   AS[(i + 11) * maxNZ + j + 5] * x[JA[(i + 11) * maxNZ + j + 5]] +
                   AS[(i + 11) * maxNZ + j + 6] * x[JA[(i + 11) * maxNZ + j + 6]] +
                   AS[(i + 11) * maxNZ + j + 7] * x[JA[(i + 11) * maxNZ + j + 7]] +
                   AS[(i + 11) * maxNZ + j + 8] * x[JA[(i + 11) * maxNZ + j + 8]] +
                   AS[(i + 11) * maxNZ + j + 9] * x[JA[(i + 11) * maxNZ + j + 9]] +
                   AS[(i + 11) * maxNZ + j + 10] * x[JA[(i + 11) * maxNZ + j + 10]] +
                   AS[(i + 11) * maxNZ + j + 11] * x[JA[(i + 11) * maxNZ + j + 11]] +
                   AS[(i + 11) * maxNZ + j + 12] * x[JA[(i + 11) * maxNZ + j + 12]] +
                   AS[(i + 11) * maxNZ + j + 13] * x[JA[(i + 11) * maxNZ + j + 13]] +
                   AS[(i + 11) * maxNZ + j + 14] * x[JA[(i + 11) * maxNZ + j + 14]] +
                   AS[(i + 11) * maxNZ + j + 15] * x[JA[(i + 11) * maxNZ + j + 15]];
            t12 += AS[(i + 12) * maxNZ + j + 0] * x[JA[(i + 12) * maxNZ + j + 0]] +
                   AS[(i + 12) * maxNZ + j + 1] * x[JA[(i + 12) * maxNZ + j + 1]] +
                   AS[(i + 12) * maxNZ + j + 2] * x[JA[(i + 12) * maxNZ + j + 2]] +
                   AS[(i + 12) * maxNZ + j + 3] * x[JA[(i + 12) * maxNZ + j + 3]] +
                   AS[(i + 12) * maxNZ + j + 4] * x[JA[(i + 12) * maxNZ + j + 4]] +
                   AS[(i + 12) * maxNZ + j + 5] * x[JA[(i + 12) * maxNZ + j + 5]] +
                   AS[(i + 12) * maxNZ + j + 6] * x[JA[(i + 12) * maxNZ + j + 6]] +
                   AS[(i + 12) * maxNZ + j + 7] * x[JA[(i + 12) * maxNZ + j + 7]] +
                   AS[(i + 12) * maxNZ + j + 8] * x[JA[(i + 12) * maxNZ + j + 8]] +
                   AS[(i + 12) * maxNZ + j + 9] * x[JA[(i + 12) * maxNZ + j + 9]] +
                   AS[(i + 12) * maxNZ + j + 10] * x[JA[(i + 12) * maxNZ + j + 10]] +
                   AS[(i + 12) * maxNZ + j + 11] * x[JA[(i + 12) * maxNZ + j + 11]] +
                   AS[(i + 12) * maxNZ + j + 12] * x[JA[(i + 12) * maxNZ + j + 12]] +
                   AS[(i + 12) * maxNZ + j + 13] * x[JA[(i + 12) * maxNZ + j + 13]] +
                   AS[(i + 12) * maxNZ + j + 14] * x[JA[(i + 12) * maxNZ + j + 14]] +
                   AS[(i + 12) * maxNZ + j + 15] * x[JA[(i + 12) * maxNZ + j + 15]];
            t13 += AS[(i + 13) * maxNZ + j + 0] * x[JA[(i + 13) * maxNZ + j + 0]] +
                   AS[(i + 13) * maxNZ + j + 1] * x[JA[(i + 13) * maxNZ + j + 1]] +
                   AS[(i + 13) * maxNZ + j + 2] * x[JA[(i + 13) * maxNZ + j + 2]] +
                   AS[(i + 13) * maxNZ + j + 3] * x[JA[(i + 13) * maxNZ + j + 3]] +
                   AS[(i + 13) * maxNZ + j + 4] * x[JA[(i + 13) * maxNZ + j + 4]] +
                   AS[(i + 13) * maxNZ + j + 5] * x[JA[(i + 13) * maxNZ + j + 5]] +
                   AS[(i + 13) * maxNZ + j + 6] * x[JA[(i + 13) * maxNZ + j + 6]] +
                   AS[(i + 13) * maxNZ + j + 7] * x[JA[(i + 13) * maxNZ + j + 7]] +
                   AS[(i + 13) * maxNZ + j + 8] * x[JA[(i + 13) * maxNZ + j + 8]] +
                   AS[(i + 13) * maxNZ + j + 9] * x[JA[(i + 13) * maxNZ + j + 9]] +
                   AS[(i + 13) * maxNZ + j + 10] * x[JA[(i + 13) * maxNZ + j + 10]] +
                   AS[(i + 13) * maxNZ + j + 11] * x[JA[(i + 13) * maxNZ + j + 11]] +
                   AS[(i + 13) * maxNZ + j + 12] * x[JA[(i + 13) * maxNZ + j + 12]] +
                   AS[(i + 13) * maxNZ + j + 13] * x[JA[(i + 13) * maxNZ + j + 13]] +
                   AS[(i + 13) * maxNZ + j + 14] * x[JA[(i + 13) * maxNZ + j + 14]] +
                   AS[(i + 13) * maxNZ + j + 15] * x[JA[(i + 13) * maxNZ + j + 15]];
            t14 += AS[(i + 14) * maxNZ + j + 0] * x[JA[(i + 14) * maxNZ + j + 0]] +
                   AS[(i + 14) * maxNZ + j + 1] * x[JA[(i + 14) * maxNZ + j + 1]] +
                   AS[(i + 14) * maxNZ + j + 2] * x[JA[(i + 14) * maxNZ + j + 2]] +
                   AS[(i + 14) * maxNZ + j + 3] * x[JA[(i + 14) * maxNZ + j + 3]] +
                   AS[(i + 14) * maxNZ + j + 4] * x[JA[(i + 14) * maxNZ + j + 4]] +
                   AS[(i + 14) * maxNZ + j + 5] * x[JA[(i + 14) * maxNZ + j + 5]] +
                   AS[(i + 14) * maxNZ + j + 6] * x[JA[(i + 14) * maxNZ + j + 6]] +
                   AS[(i + 14) * maxNZ + j + 7] * x[JA[(i + 14) * maxNZ + j + 7]] +
                   AS[(i + 14) * maxNZ + j + 8] * x[JA[(i + 14) * maxNZ + j + 8]] +
                   AS[(i + 14) * maxNZ + j + 9] * x[JA[(i + 14) * maxNZ + j + 9]] +
                   AS[(i + 14) * maxNZ + j + 10] * x[JA[(i + 14) * maxNZ + j + 10]] +
                   AS[(i + 14) * maxNZ + j + 11] * x[JA[(i + 14) * maxNZ + j + 11]] +
                   AS[(i + 14) * maxNZ + j + 12] * x[JA[(i + 14) * maxNZ + j + 12]] +
                   AS[(i + 14) * maxNZ + j + 13] * x[JA[(i + 14) * maxNZ + j + 13]] +
                   AS[(i + 14) * maxNZ + j + 14] * x[JA[(i + 14) * maxNZ + j + 14]] +
                   AS[(i + 14) * maxNZ + j + 15] * x[JA[(i + 14) * maxNZ + j + 15]];
            t15 += AS[(i + 15) * maxNZ + j + 0] * x[JA[(i + 15) * maxNZ + j + 0]] +
                   AS[(i + 15) * maxNZ + j + 1] * x[JA[(i + 15) * maxNZ + j + 1]] +
                   AS[(i + 15) * maxNZ + j + 2] * x[JA[(i + 15) * maxNZ + j + 2]] +
                   AS[(i + 15) * maxNZ + j + 3] * x[JA[(i + 15) * maxNZ + j + 3]] +
                   AS[(i + 15) * maxNZ + j + 4] * x[JA[(i + 15) * maxNZ + j + 4]] +
                   AS[(i + 15) * maxNZ + j + 5] * x[JA[(i + 15) * maxNZ + j + 5]] +
                   AS[(i + 15) * maxNZ + j + 6] * x[JA[(i + 15) * maxNZ + j + 6]] +
                   AS[(i + 15) * maxNZ + j + 7] * x[JA[(i + 15) * maxNZ + j + 7]] +
                   AS[(i + 15) * maxNZ + j + 8] * x[JA[(i + 15) * maxNZ + j + 8]] +
                   AS[(i + 15) * maxNZ + j + 9] * x[JA[(i + 15) * maxNZ + j + 9]] +
                   AS[(i + 15) * maxNZ + j + 10] * x[JA[(i + 15) * maxNZ + j + 10]] +
                   AS[(i + 15) * maxNZ + j + 11] * x[JA[(i + 15) * maxNZ + j + 11]] +
                   AS[(i + 15) * maxNZ + j + 12] * x[JA[(i + 15) * maxNZ + j + 12]] +
                   AS[(i + 15) * maxNZ + j + 13] * x[JA[(i + 15) * maxNZ + j + 13]] +
                   AS[(i + 15) * maxNZ + j + 14] * x[JA[(i + 15) * maxNZ + j + 14]] +
                   AS[(i + 15) * maxNZ + j + 15] * x[JA[(i + 15) * maxNZ + j + 15]];


        }

        for (j = maxNZ - maxNZ % 16; j < maxNZ; j++) {
            t0 += AS[(i + 0) * maxNZ + j] * x[JA[(i + 0) * maxNZ + j]];
            t1 += AS[(i + 1) * maxNZ + j] * x[JA[(i + 1) * maxNZ + j]];
            t2 += AS[(i + 2) * maxNZ + j] * x[JA[(i + 2) * maxNZ + j]];
            t3 += AS[(i + 3) * maxNZ + j] * x[JA[(i + 3) * maxNZ + j]];
            t4 += AS[(i + 4) * maxNZ + j] * x[JA[(i + 4) * maxNZ + j]];
            t5 += AS[(i + 5) * maxNZ + j] * x[JA[(i + 5) * maxNZ + j]];
            t6 += AS[(i + 6) * maxNZ + j] * x[JA[(i + 6) * maxNZ + j]];
            t7 += AS[(i + 7) * maxNZ + j] * x[JA[(i + 7) * maxNZ + j]];
            t8 += AS[(i + 8) * maxNZ + j] * x[JA[(i + 8) * maxNZ + j]];
            t9 += AS[(i + 9) * maxNZ + j] * x[JA[(i + 9) * maxNZ + j]];
            t10 += AS[(i + 10) * maxNZ + j] * x[JA[(i + 10) * maxNZ + j]];
            t11 += AS[(i + 11) * maxNZ + j] * x[JA[(i + 11) * maxNZ + j]];
            t12 += AS[(i + 12) * maxNZ + j] * x[JA[(i + 12) * maxNZ + j]];
            t13 += AS[(i + 13) * maxNZ + j] * x[JA[(i + 13) * maxNZ + j]];
            t14 += AS[(i + 14) * maxNZ + j] * x[JA[(i + 14) * maxNZ + j]];
            t15 += AS[(i + 15) * maxNZ + j] * x[JA[(i + 15) * maxNZ + j]];
        }
        y[i + 0] = t0;
        y[i + 1] = t1;
        y[i + 2] = t2;
        y[i + 3] = t3;
        y[i + 4] = t4;
        y[i + 5] = t5;
        y[i + 6] = t6;
        y[i + 7] = t7;
        y[i + 8] = t8;
        y[i + 9] = t9;
        y[i + 10] = t10;
        y[i + 11] = t11;
        y[i + 12] = t12;
        y[i + 13] = t13;
        y[i + 14] = t14;
        y[i + 15] = t15;
    }

#pragma omp parallel for shared(x, y) private(t, i, j, idx)
    for (i = rows - rows % 16; i < rows; i++) {
        t = 0.0;
        for (j = 0; j < maxNZ; j++) {
            idx = i * maxNZ + j;
            t += AS[idx] * x[JA[idx]];
        }
        y[i] = t;
    }

    return y;
}