//
// Created by Yusuf Ganiyu on 2/5/23.
//
#include "smallscale.h"
#include <vector>
#include "MatrixELLPACK.h"
#include <omp.h>

MatrixELLPACK::MatrixELLPACK(int rows, int cols, int nz, int *I, int *J, double *val) {
    this->rows = rows;
    this->cols = cols;
    this->nz = nz;
    this->I = I;
    this->J = J;
    this->val = val;

    this->maxNZ = getMaxNZ(nz, I);

    this->JA = (int *) malloc(rows * maxNZ * sizeof(int));
    this->AS = (double *) malloc(rows * maxNZ * sizeof(double));

    sortData();

    setJA(nz, I, J);

    setAS(val);

    //print JA
    printf("JA: \n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < maxNZ; j++) {
            printf("%d ", this->getJA()[i * maxNZ + j] + 1);
        }
        printf("\n");
    }

    //print AS
    printf("AS: \n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < maxNZ; j++) {
            printf("%.2f ", this->getAS()[i * maxNZ + j]);
        }
        printf("\n");
    }

}

void MatrixELLPACK::sortData(){
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



double* MatrixELLPACK::multiplyELLPack(double* x) {
    double* y = (double*) malloc(rows * sizeof(double));
    double t, t0, t1, t2, t3;
    int i, j, idx; // idx is the index of (i,j)

    // We unroll to 4 to reduce the loading time
#pragma omp parallel for shared(x,y,maxNZ, JA, AS,rows,cols) private(i,j,idx,t0, t1, t2)
    for (i = 0;i < cols - cols % 4;i += 4) {
        t0 = 0;
        t1 = 0;
        t2 = 0;
        t3 = 0;

        for (j = 0;j < maxNZ - maxNZ % 2;j += 2) {
            t0 += AS[(i + 0)*maxNZ + j + 0] * x[JA[(i + 0)*maxNZ + j + 0]] + AS[(i + 0)*maxNZ + j + 1] * x[JA[(i + 0)*maxNZ + j + 1]];
            t1 += AS[(i + 1)*maxNZ + j + 0] * x[JA[(i + 1)*maxNZ + j + 0]] + AS[(i + 1)*maxNZ + j + 1] * x[JA[(i + 1)*maxNZ + j + 1]];
            t2 += AS[(i + 2)*maxNZ + j + 0] * x[JA[(i + 2)*maxNZ + j + 0]] + AS[(i + 2)*maxNZ + j + 1] * x[JA[(i + 2)*maxNZ + j + 1]];
            t3 += AS[(i + 3)*maxNZ + j + 0] * x[JA[(i + 3)*maxNZ + j + 0]] + AS[(i + 3)*maxNZ + j + 1] * x[JA[(i + 3)*maxNZ + j + 1]];
        }

        for (j = maxNZ - maxNZ % 2;j < maxNZ;j++) {
            t0 += AS[(i + 0)*maxNZ + j] * x[JA[(i + 0)*maxNZ + j]];
            t1 += AS[(i + 1)*maxNZ + j] * x[JA[(i + 1)*maxNZ + j]];
            t2 += AS[(i + 2)*maxNZ + j] * x[JA[(i + 2)*maxNZ + j]];
            t3 += AS[(i + 3)*maxNZ + j] * x[JA[(i + 3)*maxNZ + j]];
        }
        y[i + 0] = t0;
        y[i + 1] = t1;
        y[i + 2] = t2;
        y[i + 3] = t3;
    }

    for (i = rows - rows % 4;i < rows;i++) {
        t = 0.0;
        for (j = 0;j < maxNZ;j++) {
            idx = i * maxNZ + j;
            t += AS[idx] * x[JA[idx]];
        }
        y[i] = t;
    }

    return y;
}
