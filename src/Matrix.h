//
// Created by Yusuf Ganiyu on 2/20/23.
//

#ifndef SMALLSCALE_MATRIX_H
#define SMALLSCALE_MATRIX_H


struct Matrix {
    int rows;
    int cols;
    int nz;
    int *I;
    int *J;
    double *val;
    double *x;
    int *IRP;
    int *JA;
    double *AS;
    int maxNZ;
};

#endif //SMALLSCALE_MATRIX_H
