//
// Created by Yusuf Ganiyu on 2/14/23.
//

#ifndef SMALLSCALE_MATRIXBASE_H
#define SMALLSCALE_MATRIXBASE_H

class MatrixBase {

public:
    static void sortData(int* &I_, int* &J_, double* &AS_, int nz) ;

    virtual double *serialMultiply(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll2H(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll2V(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll4H(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll4V(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll8H(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll8V(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll16H(double *x, double *y) = 0;

    virtual double *openMPMultiplyUnroll16V(double *x, double *y) = 0;
};


#endif //SMALLSCALE_MATRIXBASE_H
