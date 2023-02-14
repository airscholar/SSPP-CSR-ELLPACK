//#include "smallscale.h"
#include <vector>
#include "MatrixCSR.h"
#include <omp.h>

using namespace std;

MatrixCSR::MatrixCSR(int M, int N, int nz, int *I, int *J, double *val, double *x) {
    this->rows = M;
    this->cols = N;
    this->nz = nz;
    this->I = I;
    this->J = J;
    this->val = val;
    this->x = x;

    this->IRP = new int[this->rows + 1];
    this->JA = new int[this->nz];
    this->AS = new double[this->nz];

    printf("Starting conversion to CSR\n");
    //sort the data I, J, val
//    sortData();
//    sortData(I, J, val, 0, nz - 1);
//    printf("Data sorted\n");

    //CONVERT TO CSR
    // IRP
    setIRP(nz, I);
//    printf("IRP set\n");
    // JA
    setJA(nz, J);
//    printf("JA set\n");
    // AS
    setAS(nz, val);
//    printf("AS set\n");

//    //print IRP
//    printf("IRP: ");
//    for (int i = 0; i < rows + 1; i++) {
//        printf("%d ", this->getIRP()[i] + 1);
//    }
//    printf("\n");
//
//    //print JA
//    printf("JA: ");
//    for (int i = 0; i < nz; i++) {
//        printf("%d ", this->getJA()[i] + 1);
//    }
//    printf("\n");
//
//    //print AS
//    printf("AS: ");
//    for (int i = 0; i < nz; i++) {
//        printf("%.2f ", this->getAS()[i]);
//    }
//    printf("\n");
}

void MatrixCSR::swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

void MatrixCSR::swap(double &a, double &b) {
    double temp = a;
    a = b;
    b = temp;
}

int MatrixCSR::partition(int I[], int J[], double val[], int low, int high) {
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

void MatrixCSR::sortData(int I[], int J[], double val[], int low, int high) {
//    if (low < high) {
//        int pivotIndex = partition(I, J, val, low, high);
//        sortData(I, J, val, low, pivotIndex - 1);
//        sortData(I, J, val, pivotIndex + 1, high);
//    }
////    printf("Data sorted %d %d\n", low, high);
//}
    int i, j;
    for (i = 1; i < nz; i++) {
        int elem1 = I[i];
        int elem2 = J[i];
        double elem3 = val[i];
        for (j = i; j > 0 && I[j - 1] > elem1; j--) {
            I[j] = I[j - 1];
            J[j] = J[j - 1];
            val[j] = val[j - 1];
        }
        I[j] = elem1;
        J[j] = elem2;
        val[j] = elem3;
    }
}

void MatrixCSR::setJA(int nz, int *J) {
    for (int i = 0; i < nz; i++) {
        this->JA[i] = J[i];
    }
}

void MatrixCSR::setAS(int nz, double *val) {
    for (int i = 0; i < nz; i++) {
        AS[i] = val[i];
    }
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

int *MatrixCSR::getIRP() {
    return this->IRP;
}

int *MatrixCSR::getJA() {
    return this->JA;
}

double *MatrixCSR::getAS() {
    return this->AS;
}

double *MatrixCSR::serialMultiply(double *v, double *y) {
#pragma omp parallel for shared(y, v)
    for (int i = 0; i < rows; i++) {
        double t = 0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            t += AS[j] * v[JA[j]];
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
            for (j = IRP[k]; j < IRP[k + 1] - 15; j += 16) {
                t0 += AS[j] * v[JA[j]];
                t1 += AS[j + 1] * v[JA[j + 1]];
                t2 += AS[j + 2] * v[JA[j + 2]];
                t3 += AS[j + 3] * v[JA[j + 3]];
                t4 += AS[j + 4] * v[JA[j + 4]];
                t5 += AS[j + 5] * v[JA[j + 5]];
                t6 += AS[j + 6] * v[JA[j + 6]];
                t7 += AS[j + 7] * v[JA[j + 7]];
                t8 += AS[j + 8] * v[JA[j + 8]];
                t9 += AS[j + 9] * v[JA[j + 9]];
                t10 += AS[j + 10] * v[JA[j + 10]];
                t11 += AS[j + 11] * v[JA[j + 11]];
                t12 += AS[j + 12] * v[JA[j + 12]];
                t13 += AS[j + 13] * v[JA[j + 13]];
                t14 += AS[j + 14] * v[JA[j + 14]];
                t15 += AS[j + 15] * v[JA[j + 15]];
            }
            for (; j < IRP[k + 1]; j++) {
                t0 += AS[j] * v[JA[j]];
            }
            y[k] = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t14 + t15;
//        y[i] = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t14 + t15;
        }
    }
    return y;
}
