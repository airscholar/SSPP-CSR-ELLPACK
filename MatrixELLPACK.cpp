//
// Created by Yusuf Ganiyu on 2/5/23.
//

#include "MatrixELLPACK.h"

MatrixELLPACK::MatrixELLPACK(int rows, int cols, int nz, int *I, int *J, double *val) {
    this->rows = rows;
    this->cols = cols;
    this->nz = nz;
    this->I = I;
    this->J = J;

    this->JA.resize(rows);
    this->AS.resize(rows);

    int *lPackNzTab = (int *) malloc(rows * sizeof(int));

    // fill the lPackNzTab with zeros
    for (int i = 0; i < rows; i++) {
        lPackNzTab[i] = 0;
    }

    maxNZ = lPackNzTab[0];
    // fill the lPackNzTab with the number of non-zero elements in each row
    for (int i = 1; i < rows; i++) {
        if (lPackNzTab[i] > maxNZ) {
            maxNZ = lPackNzTab[i];
        }
    }

    // fill the JA and AS vectors with the zeros
    for (int i = 0; i < rows; i++) {
        JA[i].resize(maxNZ, 0);
        AS[i].resize(maxNZ, 0);
    }

    // fill the JA and AS vectors with the non-zero elements
    for (int i = 0; i < nz; i++) {
        JA[I[i]].push_back(J[i]);
        AS[I[i]].push_back(val[i]);
    }

    //resize the JA and AS vectors to the maxNZ
    for (int i = 0; i < rows; i++) {
        if (JA[i].size() < maxNZ) {
            JA[i].resize(maxNZ, JA[i][JA[i].size()]);
            printf("JA size: %d", JA[i].size());
        }
        if (AS[i].size() < maxNZ) {
            AS[i].resize(maxNZ, 0);
        }
    }

    printf("JA: \n");
    for (int i = 0; i < JA.size(); i++) {
        for (int j = 0; j < JA[i].size(); j++) {
            printf("%d ", JA[i][j]);
        }
        printf("\n");
    }

    printf("AS: \n");
    for (int i = 0; i < AS.size(); i++) {
        for (int j = 0; j < AS[i].size(); j++) {
            printf("%.2f ", AS[i][j]);
        }
        printf("\n");
    }

    // multiply the matrix
//    this->multiply(rows, maxNZ, JA, AS, val);
}

std::vector<std::vector<int>> MatrixELLPACK::getJA() {
    return this->JA;
}

std::vector<std::vector<double>> MatrixELLPACK::getAS() {
    return this->AS;
}
//std::vector<int> MatrixELLPACK::multiply(int rows, int maxNZ, std::vector<std::vector<int>> JA, std::vector<std::vector<double>> AS, std::vector<double> x) {
//    std::vector<int> result(rows, 0);
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < maxNZ; j++) {
//            result[i] += AS[i][j] * x[JA[i][j]];
//        }
//    }
//    return result;
//}
