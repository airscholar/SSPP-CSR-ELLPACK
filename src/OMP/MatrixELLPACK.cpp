//
// Created by Yusuf Ganiyu on 2/5/23.
//
#include <vector>
#include "MatrixELLPACK.h"

// Returns the MAXNZ of the ELLPACK
int MatrixELLPACK::getMaxNZ(int nz, int *I) {
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

void MatrixELLPACK::setJA(int nz, int *I, int *J) {
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

void MatrixELLPACK::setAS(int maxNZ, double *val) {
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
        }
    }
}

double* MatrixELLPACK::getAS(){
    return this->AS;
}

int* MatrixELLPACK::getJA(){
    return this->JA;
}

//SERIAL
double* MatrixELLPACK::serialMultiply(double* x, double* y) {

    // Iterate through all the rows of the matrix.
#pragma omp parallel for shared(x, y)
    for (int i = 0; i < rows; i++) {
        double t = 0;
        // Iterate through all the non-zero elements of the current row.
        for (int j = 0; j < maxNZ; j++) {
            // If the current element is the last NZ element of the row, then break out of the loop.
            if (JA[i * maxNZ + j] == -1) {
                break;
            }
            // Accumulate the product of the current non-zero element and the corresponding entry in the input vector.
            t += AS[i * maxNZ + j] * x[JA[i * maxNZ + j]];
        }
    }

    return y;
}

//VERTICAL UNROLLING
double* MatrixELLPACK::openMPMultiplyUnroll2V(double *x, double *y) {
    double t, t0, t1;
    int i, j, idx; // idx is the index of (i,j)

#pragma omp parallel for shared(x, y) private(t, t0, t1, i, j, idx)
    for (i = 0; i < rows - rows % 2; i += 2) {
        t0 = 0, t1 = 0;
        for (j = 0; j < maxNZ; j++) {
            if (JA[i * maxNZ + j] == -1 && JA[(i + 1) * maxNZ + j] == -1) {
                break;
            }
            if ((i + 0) * maxNZ + j < rows)
                t0 += AS[i * maxNZ + j] * x[JA[i * maxNZ + j]];
            if ((i + 1) * maxNZ + j < rows)
                t1 += AS[(i + 1) * maxNZ + j] * x[JA[(i + 1) * maxNZ + j]];
        }
        y[i] = t0;
        y[i + 1] = t1;
    }

    //handle the rest
    for (i = rows - rows % 2; i < rows; i++) {
        t = 0;
        for (j = 0; j < maxNZ; j++) {
            if (JA[i * maxNZ + j] == -1) {
                break;
            }
            t += AS[i * maxNZ + j] * x[JA[i * maxNZ + j]];
        }
        y[i] = t;
    }

    return y;
}

double* MatrixELLPACK::openMPMultiplyUnroll4V(double *x, double *y) {
    double t, t0, t1, t2, t3;
    int i, j, idx; // idx is the index of (i,j)

#pragma omp parallel for shared(x, y) private(t, t0, t1, t2, t3, i, j, idx)
    for (i = 0; i < rows - rows % 4; i += 4) {
        t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (j = 0; j < maxNZ; j++) {
            if (JA[i * maxNZ + j] == -1 && JA[(i + 1) * maxNZ + j] == -1 && JA[(i + 2) * maxNZ + j] == -1 &&
                JA[(i + 3) * maxNZ + j] == -1) {
                break;
            }
            if ((i + 0) * maxNZ + j < rows)
                t0 += AS[i * maxNZ + j] * x[JA[i * maxNZ + j]];
            if ((i + 1) * maxNZ + j < rows)
                t1 += AS[(i + 1) * maxNZ + j] * x[JA[(i + 1) * maxNZ + j]];
            if ((i + 2) * maxNZ + j < rows)
                t2 += AS[(i + 2) * maxNZ + j] * x[JA[(i + 2) * maxNZ + j]];
            if ((i + 3) * maxNZ + j < rows)
                t3 += AS[(i + 3) * maxNZ + j] * x[JA[(i + 3) * maxNZ + j]];
        }
        y[i] = t0;
        y[i + 1] = t1;
        y[i + 2] = t2;
        y[i + 3] = t3;
    }

    //handle the rest
    for (i = rows - rows % 4; i < rows; i++) {
        t = 0;
        for (j = 0; j < maxNZ; j++) {
            if (JA[i * maxNZ + j] == -1) {
                break;
            }
            t += AS[i * maxNZ + j] * x[JA[i * maxNZ + j]];
        }
        y[i] = t;
    }

    return y;
}

double* MatrixELLPACK::openMPMultiplyUnroll8V(double *x, double *y) {
    double t, t0, t1, t2, t3, t4, t5, t6, t7;
    int i, j, idx; // idx is the index of (i,j)

#pragma omp parallel for shared(x, y) private(t, t0, t1, t2, t3, t4, t5, t6, t7, i, j, idx)
    for (i = 0; i < rows - rows % 8; i += 8) {
        t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
        for (j = 0; j < maxNZ; j++) {
            if (JA[i * maxNZ + j] == -1 && JA[(i + 1) * maxNZ + j] == -1 && JA[(i + 2) * maxNZ + j] == -1 &&
                JA[(i + 3) * maxNZ + j] == -1 &&
                JA[(i + 4) * maxNZ + j] == -1 && JA[(i + 5) * maxNZ + j] == -1 && JA[(i + 6) * maxNZ + j] == -1 &&
                JA[(i + 7) * maxNZ + j] == -1) {
                break;
            }

            if ((i + 0) * maxNZ + j < rows)
                t0 += AS[i * maxNZ + j] * x[JA[i * maxNZ + j]];
            if ((i + 1) * maxNZ + j < rows)
                t1 += AS[(i + 1) * maxNZ + j] * x[JA[(i + 1) * maxNZ + j]];
            if ((i + 2) * maxNZ + j < rows)
                t2 += AS[(i + 2) * maxNZ + j] * x[JA[(i + 2) * maxNZ + j]];
            if ((i + 3) * maxNZ + j < rows)
                t3 += AS[(i + 3) * maxNZ + j] * x[JA[(i + 3) * maxNZ + j]];
            if ((i + 4) * maxNZ + j < rows)
                t4 += AS[(i + 4) * maxNZ + j] * x[JA[(i + 4) * maxNZ + j]];
            if ((i + 5) * maxNZ + j < rows)
                t5 += AS[(i + 5) * maxNZ + j] * x[JA[(i + 5) * maxNZ + j]];
            if ((i + 6) * maxNZ + j < rows)
                t6 += AS[(i + 6) * maxNZ + j] * x[JA[(i + 6) * maxNZ + j]];
            if ((i + 7) * maxNZ + j < rows)
                t7 += AS[(i + 7) * maxNZ + j] * x[JA[(i + 7) * maxNZ + j]];
        }
        y[i] = t0;
        y[i + 1] = t1;
        y[i + 2] = t2;
        y[i + 3] = t3;
        y[i + 4] = t4;
        y[i + 5] = t5;
        y[i + 6] = t6;
        y[i + 7] = t7;
    }

    //handle the rest
    for (i = rows - rows % 8; i < rows; i++) {
        t = 0;
        for (j = 0; j < maxNZ; j++) {
            if (JA[i * maxNZ + j] == -1) {
                break;
            }
            t += AS[i * maxNZ + j] * x[JA[i * maxNZ + j]];
        }
        y[i] = t;
    }

    return y;
}

double* MatrixELLPACK::openMPMultiplyUnroll16V(double *x, double *y) {
    double t, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    int i, j, idx; // idx is the index of (i,j)

#pragma omp parallel for shared(x, y) private(t, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, i, j, idx)
    for (i = 0; i < rows - rows % 16; i += 16) {
        t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0,
        t14 = 0, t15 = 0;
        for (j = 0; j < maxNZ; j++) {
            if (JA[i * maxNZ + j] == -1 && JA[(i + 1) * maxNZ + j] == -1 && JA[(i + 2) * maxNZ + j] == -1 &&
                JA[(i + 3) * maxNZ + j] == -1 && JA[(i + 4) * maxNZ + j] == -1 && JA[(i + 5) * maxNZ + j] == -1 &&
                JA[(i + 6) * maxNZ + j] == -1 && JA[(i + 7) * maxNZ + j] == -1 && JA[(i + 8) * maxNZ + j] == -1 &&
                JA[(i + 9) * maxNZ + j] == -1 && JA[(i + 10) * maxNZ + j] == -1 && JA[(i + 11) * maxNZ + j] == -1 &&
                JA[(i + 12) * maxNZ + j] == -1 && JA[(i + 13) * maxNZ + j] == -1 && JA[(i + 14) * maxNZ + j] == -1 &&
                JA[(i + 15) * maxNZ + j] == -1) {
                break;
            }

            if ((i + 0) * maxNZ + j < rows)
                t0 += AS[i * maxNZ + j] * x[JA[i * maxNZ + j]];
            if ((i + 1) * maxNZ + j < rows)
                t1 += AS[(i + 1) * maxNZ + j] * x[JA[(i + 1) * maxNZ + j]];
            if ((i + 2) * maxNZ + j < rows)
                t2 += AS[(i + 2) * maxNZ + j] * x[JA[(i + 2) * maxNZ + j]];
            if ((i + 3) * maxNZ + j < rows)
                t3 += AS[(i + 3) * maxNZ + j] * x[JA[(i + 3) * maxNZ + j]];
            if ((i + 4) * maxNZ + j < rows)
                t4 += AS[(i + 4) * maxNZ + j] * x[JA[(i + 4) * maxNZ + j]];
            if ((i + 5) * maxNZ + j < rows)
                t5 += AS[(i + 5) * maxNZ + j] * x[JA[(i + 5) * maxNZ + j]];
            if ((i + 6) * maxNZ + j < rows)
                t6 += AS[(i + 6) * maxNZ + j] * x[JA[(i + 6) * maxNZ + j]];
            if ((i + 7) * maxNZ + j < rows)
                t7 += AS[(i + 7) * maxNZ + j] * x[JA[(i + 7) * maxNZ + j]];
            if ((i + 8) * maxNZ + j < rows)
                t8 += AS[(i + 8) * maxNZ + j] * x[JA[(i + 8) * maxNZ + j]];
            if ((i + 9) * maxNZ + j < rows)
                t9 += AS[(i + 9) * maxNZ + j] * x[JA[(i + 9) * maxNZ + j]];
            if ((i + 10) * maxNZ + j < rows)
                t10 += AS[(i + 10) * maxNZ + j] * x[JA[(i + 10) * maxNZ + j]];
            if ((i + 11) * maxNZ + j < rows)
                t11 += AS[(i + 11) * maxNZ + j] * x[JA[(i + 11) * maxNZ + j]];
            if ((i + 12) * maxNZ + j < rows)
                t12 += AS[(i + 12) * maxNZ + j] * x[JA[(i + 12) * maxNZ + j]];
            if ((i + 13) * maxNZ + j < rows)
                t13 += AS[(i + 13) * maxNZ + j] * x[JA[(i + 13) * maxNZ + j]];
            if ((i + 14) * maxNZ + j < rows)
                t14 += AS[(i + 14) * maxNZ + j] * x[JA[(i + 14) * maxNZ + j]];
            if ((i + 15) * maxNZ + j < rows)
                t15 += AS[(i + 15) * maxNZ + j] * x[JA[(i + 15) * maxNZ + j]];
        }

        y[i] = t0;
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

    //handle the rest
    for (i = rows - rows % 16; i < rows; i++) {
        t = 0;
        for (j = 0; j < maxNZ; j++) {
            if (JA[i * maxNZ + j] == -1) {
                break;
            }
            t += AS[i * maxNZ + j] * x[JA[i * maxNZ + j]];
        }
        y[i] = t;
    }

    return y;
}

// HORIZONTAL UNROLLING
double* MatrixELLPACK::openMPMultiplyUnroll2H(double *x, double *y) {
#pragma omp parallel for shared(x, y)
    for (int row = 0; row < rows; row++) {
        double sum = 0;
        //unroll column
        for (int col = 0; col < maxNZ - maxNZ % 2; col += 2) {
            sum += AS[row * maxNZ + col] * x[JA[row * maxNZ + col]];
            sum += AS[row * maxNZ + col + 1] * x[JA[row * maxNZ + col + 1]];
        }

        //handle the rest
        for (int col = maxNZ - maxNZ % 2; col < maxNZ; col++) {
            sum += AS[row * maxNZ + col] * x[JA[row * maxNZ + col]];
        }

        y[row] = sum;
    }

    return y;
}

double* MatrixELLPACK::openMPMultiplyUnroll4H(double *x, double *y) {
    double t, t0, t1, t2, t3;
    int i, j, idx; // idx is the index of (i,j)

    #pragma omp parallel for shared(x, y) private(t, t0, t1, t2, t3, i, j, idx)
    for (int row = 0; row < rows; row++) {
        double sum = 0;
        //unroll column
        for (int col = 0; col < maxNZ - maxNZ % 4; col += 4) {
            if (row * maxNZ + col < rows)
                sum += AS[row * maxNZ + col] * x[JA[row * maxNZ + col]];
            if (row * maxNZ + col + 1 < rows)
                sum += AS[row * maxNZ + col + 1] * x[JA[row * maxNZ + col + 1]];
            if (row * maxNZ + col + 2 < rows)
                sum += AS[row * maxNZ + col + 2] * x[JA[row * maxNZ + col + 2]];
            if (row * maxNZ + col + 3 < rows)
                sum += AS[row * maxNZ + col + 3] * x[JA[row * maxNZ + col + 3]];
        }

        //handle the rest
        for (int col = maxNZ - maxNZ % 4; col < maxNZ; col++) {
            sum += AS[row * maxNZ + col] * x[JA[row * maxNZ + col]];
        }

        y[row] = sum;
    }

    return y;
}

double* MatrixELLPACK::openMPMultiplyUnroll8H(double *x, double *y) {
    //openmp multiplication of ELLPACK matrix UNROLL 8
    double t, t0, t1, t2, t3, t4, t5, t6, t7;
    int i, j, k;

    #pragma omp parallel for private(i, j, k, t, t0, t1, t2, t3, t4, t5, t6, t7)
    for (int row = 0; row < rows; row++) {
        double sum = 0;

        //unroll column
        for (int col = 0; col < maxNZ - maxNZ % 8; col += 8) {
            if (row * maxNZ + col < rows)
                sum += AS[row * maxNZ + col] * x[JA[row * maxNZ + col]];
            if (row * maxNZ + col + 1 < rows)
                sum += AS[row * maxNZ + col + 1] * x[JA[row * maxNZ + col + 1]];
            if (row * maxNZ + col + 2 < rows)
                sum += AS[row * maxNZ + col + 2] * x[JA[row * maxNZ + col + 2]];
            if (row * maxNZ + col + 3 < rows)
                sum += AS[row * maxNZ + col + 3] * x[JA[row * maxNZ + col + 3]];
            if (row * maxNZ + col + 4 < rows)
                sum += AS[row * maxNZ + col + 4] * x[JA[row * maxNZ + col + 4]];
            if (row * maxNZ + col + 5 < rows)
                sum += AS[row * maxNZ + col + 5] * x[JA[row * maxNZ + col + 5]];
            if (row * maxNZ + col + 6 < rows)
                sum += AS[row * maxNZ + col + 6] * x[JA[row * maxNZ + col + 6]];
            if (row * maxNZ + col + 7 < rows)
                sum += AS[row * maxNZ + col + 7] * x[JA[row * maxNZ + col + 7]];
        }

        //handle remaining columns
        for (int col = maxNZ - maxNZ % 8; col < maxNZ; col++) {
            sum += AS[row * maxNZ + col] * x[JA[row * maxNZ + col]];
        }

        y[row] = sum;
    }

    return y;
}

double* MatrixELLPACK::openMPMultiplyUnroll16H(double *x, double *y) {
//openmp multiplication of ELLPACK matrix UNROLL 16
    double t, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    int i, j, k;

#pragma omp parallel for private(i, j, k, t, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15)
    for (int row = 0; row < rows; row++) {
        double sum = 0;

        //unroll column
        for (int col = 0; col < maxNZ - maxNZ % 16; col += 16) {
            if (row * maxNZ + col < rows)
                sum += AS[row * maxNZ + col] * x[JA[row * maxNZ + col]];
            if (row * maxNZ + col + 1 < rows)
                sum += AS[row * maxNZ + col + 1] * x[JA[row * maxNZ + col + 1]];
            if (row * maxNZ + col + 2 < rows)
                sum += AS[row * maxNZ + col + 2] * x[JA[row * maxNZ + col + 2]];
            if (row * maxNZ + col + 3 < rows)
                sum += AS[row * maxNZ + col + 3] * x[JA[row * maxNZ + col + 3]];
            if (row * maxNZ + col + 4 < rows)
                sum += AS[row * maxNZ + col + 4] * x[JA[row * maxNZ + col + 4]];
            if (row * maxNZ + col + 5 < rows)
                sum += AS[row * maxNZ + col + 5] * x[JA[row * maxNZ + col + 5]];
            if (row * maxNZ + col + 6 < rows)
                sum += AS[row * maxNZ + col + 6] * x[JA[row * maxNZ + col + 6]];
            if (row * maxNZ + col + 7 < rows)
                sum += AS[row * maxNZ + col + 7] * x[JA[row * maxNZ + col + 7]];
            if (row * maxNZ + col + 8 < rows)
                sum += AS[row * maxNZ + col + 8] * x[JA[row * maxNZ + col + 8]];
            if (row * maxNZ + col + 9 < rows)
                sum += AS[row * maxNZ + col + 9] * x[JA[row * maxNZ + col + 9]];
            if (row * maxNZ + col + 10 < rows)
                sum += AS[row * maxNZ + col + 10] * x[JA[row * maxNZ + col + 10]];
            if (row * maxNZ + col + 11 < rows)
                sum += AS[row * maxNZ + col + 11] * x[JA[row * maxNZ + col + 11]];
            if (row * maxNZ + col + 12 < rows)
                sum += AS[row * maxNZ + col + 12] * x[JA[row * maxNZ + col + 12]];
            if (row * maxNZ + col + 13 < rows)
                sum += AS[row * maxNZ + col + 13] * x[JA[row * maxNZ + col + 13]];
            if (row * maxNZ + col + 14 < rows)
                sum += AS[row * maxNZ + col + 14] * x[JA[row * maxNZ + col + 14]];
            if (row * maxNZ + col + 15 < rows)
                sum += AS[row * maxNZ + col + 15] * x[JA[row * maxNZ + col + 15]];
        }

        //remainder
        for (int col = maxNZ - maxNZ % 16; col < maxNZ; col++) {
            sum += AS[row * maxNZ + col] * x[JA[row * maxNZ + col]];
        }

        y[row] = sum;
    }

    return y;
}