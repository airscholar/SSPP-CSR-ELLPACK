#include "smallscale.h"
#include <iostream>
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

    // IRP
    this->IRP.resize(this->rows + 1, 0);

    //sort by row
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

    // IRP
    IRP[0] = 0;
    int minJ = 0;
    for ( i = 1; i < rows + 1; i++) {
        int n = 0;
        for ( j = minJ; I[j] == i - 1; j++) {
            n++;
        }
        minJ = j;
        IRP[i] = IRP[i - 1] + n;
    }

    // AS
    this->AS.assign(val, val + nz);

    // JA
    this->JA.assign(J, J + nz);

    printf("IRP: ");
    for (int i = 0; i < IRP.size(); i++) {
        printf("%d ", IRP[i]);
    }
    printf("\n");

    printf("JA: ");
    for (int i = 0; i < JA.size(); i++) {
        printf("%d ", JA[i]);
    }
    printf("\n");

    printf("AS: ");
    for (int i = 0; i < AS.size(); i++) {
        printf("%.2f ", AS[i]);
    }
    printf("\n");

    // multiply
    vector<int> multRes = this->serialMultiply(IRP);
    vector<int> openMPres = this->openMPMultiply(IRP);

    ::printf("Result: ");
    for (int i = 0; i < multRes.size(); i++) {
        printf("%d ", multRes[i]);
    }
    printf("\n");

    printf("Parallel result: ");
    for (int i = 0; i < openMPres.size(); i++) {
        printf("%d ", openMPres[i]);
    }
    printf("\n");

}

vector<int> MatrixCSR::getIRP() {
    return this->IRP;
}

vector<int> MatrixCSR::getJA() {
    return this->JA;
}

vector<double> MatrixCSR::getAS() {
    return this->AS;
}

vector<int> MatrixCSR::serialMultiply(const vector<int> v) {
    vector<int> result(this->rows, 0);
    for (int i = 0; i < this->rows; i++) {
        for (int j = this->IRP[i]; j < this->IRP[i + 1]; j++) {
            result[i] += this->AS[j] * v[this->JA[j]];
        }
    }
    return result;
}

vector<int> MatrixCSR::openMPMultiply(const vector<int> v) {
    vector<int> result(this->rows, 0);
#pragma omp parallel for
    for (int i = 0; i < this->rows; i++) {
        for (int j = this->IRP[i]; j < this->IRP[i + 1]; j++) {
            result[i] += this->AS[j] * v[this->JA[j]];
        }
    }
    return result;
}