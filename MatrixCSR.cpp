#include "smallscale.h"
#include <vector>
#include "MatrixCSR.h"
#include <omp.h>

using namespace std;

MatrixCSR::MatrixCSR(int M, int N, int nz, int *I, int *J, double *val) {
    this->rows = M;
    this->cols = N;
    this->nz = nz;
    this->I = I;
    this->J = J;
    this->val = val;

    //sort the data I, J, val
    sortData();

    // IRP
    setIRP(nz, I);
    // JA
    setJA(nz, J);

    // AS
    setAS(nz, val);

    //print IRP
    printf("IRP: ");
    for (int i = 0; i < rows + 1; i++) {
        printf("%d ", this->getIRP()[i] + 1);
    }
    printf("\n");

    //print JA
    printf("JA: ");
    for (int i = 0; i < nz; i++) {
        printf("%d ", this->getJA()[i] + 1);
    }
    printf("\n");

    //print AS
    printf("AS: ");
    for (int i = 0; i < nz; i++) {
        printf("%.2f ", this->getAS()[i]);
    }
    printf("\n");

    // multiply
    double *multRes = this->serialMultiply(this->getIRP());
    double *openMPres = this->openMPMultiply(this->getIRP());

    printf("Result: ");
    for (int i = 0; i < this->rows; i++) {
        printf("%.2f ", multRes[i]);
    }
    printf("\n");

    printf("Parallel result: ");
    for (int i = 0; i < this->rows; i++) {
        printf("%.2f ", openMPres[i]);
    }
    printf("\n");

    //find the difference
    float diff = 0;
    for (int i = 0; i < this->rows; i++) {
        diff += abs((float) multRes[i] - (float) openMPres[i]);
    }

    printf("Validation result: %.9f\n", diff);
}

void MatrixCSR::sortData() {
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
        JA[i] = J[i];
    }
}

void MatrixCSR::setAS(int nz, double *val) {
    this->AS = new double[nz];
    for (int i = 0; i < nz; i++) {
        AS[i] = val[i];
    }
}

void MatrixCSR::setIRP(int nz, int *I) {
    this->IRP = new int[rows + 1];
    // Here we use I once it is reordered
    int k = 1; // k will be the current index+1 of the IRP array
    int i; // i will be the current index+1 of the I array
    int irp = 0; // irp will correspond to the value of IRP

    // We put the first values to zero if the first rows are empty
    while (I[0] + 1 != k) {
        IRP[k - 1] = 0;
        k++;
    }

    // Now I[0]+1 == k
    IRP[k - 1] = irp; // The first value is always zero in C
    k++;
    irp++;

    // We go through the I array
    for (i = 2; i <= nz; i++) {
        // If we are on a new row, we can put a new value in IRP
        if (I[i - 1] == I[i - 2] + 1) {
            IRP[k - 1] = irp;
            k++;
            irp++;
        } // We have skipped at least a row
        else if (I[i - 1] > I[i - 2] + 1) {
            // We need to input the previous value again as many times as there are skipped rows
            for (int skipped = 1; skipped <= I[i - 1] - I[i - 2] - 1; skipped++) {
                IRP[k - 1] = irp;
                k++;
            }
            // We also need to input the value corresponding to the new row
            IRP[k - 1] = irp;
            k++;
            irp++;
        } else {
            // The value increases because we have stayed on the same row but are moving in the index of non zero values
            irp++;
        }
    }

    // The last value is the number of non zero values in C
    IRP[rows] = nz;
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

double* MatrixCSR::serialMultiply(int* v) {
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

double* MatrixCSR::openMPMultiply(int* x) {
    double *y = new double[this->rows];
    double temp;
    int i, j;
    // We try to unroll to 4 if possible
#pragma omp parallel for shared(x, IRP, JA, AS) private(i, j, temp)
    for (int i = 0; i < rows; i++) {
        temp = 0;
        if (IRP[i + 1] - IRP[i] >= 4) {
            for (int j = IRP[i]; j <= (IRP[i + 1] - 1) - (IRP[i + 1] - 1) % 4; j += 4) {
                temp += AS[j] * x[JA[j]] + AS[j + 1] * x[JA[j + 1]] + AS[j + 2] * x[JA[j + 2]] +
                        AS[j + 3] * x[JA[j + 3]];
            }
            for (int j = (IRP[i + 1] - 1) - (IRP[i + 1] - 1) % 4; j <= IRP[i + 1] - 1; j++) {
                temp += AS[j] * x[JA[j]];
            }
        } else {
            for (int j = IRP[i]; j <= IRP[i + 1] - 1; j++) {
                temp += AS[j] * x[JA[j]];
            }
        }
        y[i] = temp;

    }
    return y;
}