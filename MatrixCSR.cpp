#include "smallscale.h"
#include <vector>
#include "MatrixCSR.h"
#include <omp.h>

using namespace std;

MatrixCSR::MatrixCSR(int M, int N, int nz, int *I, int *J, double *val, double* x) {
    this->rows = M;
    this->cols = N;
    this->nz = nz;
    this->I = I;
    this->J = J;
    this->val = val;
    this->x = x;

    printf("Starting conversion to CSR\n");
    //sort the data I, J, val
    sortData(I, J, val, 0, nz - 1);
    printf("Data sorted\n");

    //CONVERT TO CSR
    // IRP
    this->setIRP(nz, I);
    printf("IRP set\n");
    // JA
    setJA(nz, J);
    printf("JA set\n");
    // AS
    setAS(nz, val);
    printf("AS set\n");

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
//    printf("Data sorted %d %d\n", low, high);
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
    this->JA = new int[nz];
    for (int i = 0; i < nz; i++) {
        this->JA[i] = J[i];
    }
}

void MatrixCSR::setAS(int nz, double *val) {
    this->AS = new double[nz];
    for (int i = 0; i < nz; i++) {
        AS[i] = val[i];
    }
}

void MatrixCSR::setIRP(int nz, int *I) {
    this->IRP = new int[this->rows + 1];
    // Here we use I once it is reordered
    int k = 1; // k will be the current index+1 of the IRP array
    int i; // i will be the current index+1 of the I array
    int irp = 0; // irp will correspond to the value of IRP

    // We put the first values to zero if the first rows are empty
    while (I[0] + 1 != k) {
        this->IRP[k - 1] = 0;
        k++;
    }

    // Now I[0]+1 == k
    this->IRP[k - 1] = irp; // The first value is always zero in C
    k++;
    irp++;

    // We go through the I array
    for (i = 2; i <= nz; i++) {
        // If we are on a new row, we can put a new value in IRP
        if (I[i - 1] == I[i - 2] + 1) {
            this->IRP[k - 1] = irp;
            k++;
            irp++;
        } // We have skipped at least a row
        else if (I[i - 1] > I[i - 2] + 1) {
            // We need to input the previous value again as many times as there are skipped rows
            for (int skipped = 1; skipped <= I[i - 1] - I[i - 2] - 1; skipped++) {
                this->IRP[k - 1] = irp;
                k++;
            }
            // We also need to input the value corresponding to the new row
            this->IRP[k - 1] = irp;
            k++;
            irp++;
        } else {
            // The value increases because we have stayed on the same row but are moving in the index of non zero values
            irp++;
        }
    }

    // The last value is the number of non zero values in C
    this->IRP[rows] = nz;
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

double* MatrixCSR::serialMultiply(double* v) {
    double* y = new double[rows];
    double t;
    int i, j;
    for (i = 0; i < rows; i++) {
        t = 0;
        for (j = IRP[i]; j < IRP[i + 1]; j++) {
            t += AS[j] * v[JA[j]];
        }
        y[i] = t;
    }

    return y;
}

double* MatrixCSR::openMPMultiply(double* v) {
    double *y = new double[rows];
    double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    int i, j;
#pragma omp parallel for private(i, j, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15) shared(y, v)
    for (i = 0; i < rows; i++) {
        t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0, t14 = 0, t15 = 0;
        for (j = IRP[i]; j < IRP[i + 1] - 15; j += 16) {
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
        for (; j < IRP[i + 1]; j++) {
            t0 += AS[j] * v[JA[j]];
        }
        y[i] = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t14 + t15;
    }
    return y;
}
