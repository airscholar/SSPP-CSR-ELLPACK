//
// Created by Yusuf Ganiyu on 2/14/23.
//

#ifndef SMALLSCALE_MATRIXBASE_H
#define SMALLSCALE_MATRIXBASE_H


class MatrixBase {
private:
    int rows;
    int cols;
    int nz;
    int maxNZ;
    int *I;
    int *J;
    int **JD;
    double *val;
    double *x;
    double *y;
    int *IRP;
    int *JA;
    double *AS;

public:
    MatrixBase(int rows, int cols, int nz, int *I, int *J, double *val, double *x);
    MatrixBase(int rows, int cols, int nz, int *I, int **J, double *val, double *x);

    virtual void sortData(int *I_, int *J_, double *AS_, int nz) ;

    virtual double *serialMultiply(double *x, double *y) = 0;
    virtual double *openMPMultiply(double *x, double *y) = 0;

};


#endif //SMALLSCALE_MATRIXBASE_H
