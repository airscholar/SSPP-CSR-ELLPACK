#include <vector>
#include "MatrixCSR.h"

double *MatrixCSR::serialMultiply(double *v, double *y) {
#pragma omp parallel for shared(y, v)
    for (int i = 0; i < this->rows; i++) {
        double t = 0;
        for (int j = this->getIRP()[i]; j < this->getIRP()[i + 1]; j++) {
            t += this->getAS()[j] * v[this->getJA()[j]];
        }
        y[i] = t;
    }

    return y;
}

double *MatrixCSR::openMPMultiplyUnroll2V(double *x, double *y) {
    if (x == nullptr || y == nullptr) {
        std::cerr << "Error: x or y pointer is null" << std::endl;
        return nullptr;
    }

    if (this->cols != this->rows) {
        std::cerr << "Error: Matrix dimensions are not valid" << std::endl;
        return nullptr;
    }

    int row,  idx;

#pragma omp parallel for private(row,  idx) shared(y, x)
    for (row = 0; row < this->rows - rows % 2; row += 2) {
        double t0 = 0, t1 = 0;
        for (idx = this->getIRP()[row]; idx < this->getIRP()[row + 1]; idx++) {
            t0 += this->getAS()[idx] * x[this->getJA()[idx]];
        }
        if (row + 1 < this->rows)
            for (idx = this->getIRP()[row + 1]; idx < this->getIRP()[row + 2]; idx++) {
                t1 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        y[row] = t0;
        y[row + 1] = t1;
    }

    return y;
}

double *MatrixCSR::openMPMultiplyUnroll4V(double *x, double *y) {
    if (x == nullptr || y == nullptr) {
        std::cerr << "Error: x or y pointer is null" << std::endl;
        return nullptr;
    }

    if (this->cols != this->rows) {
        std::cerr << "Error: Matrix dimensions are not valid" << std::endl;
        return nullptr;
    }

    int row,  idx;
#pragma omp parallel for private(row,  idx) shared(y, x)
    for (row = 0; row < this->rows - rows % 4; row += 4) {
        double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (idx = this->getIRP()[row]; idx < this->getIRP()[row + 1]; idx++) {
            t0 += this->getAS()[idx] * x[this->getJA()[idx]];
        }
        if (row + 1 < this->rows)
            for (idx = this->getIRP()[row + 1]; idx < this->getIRP()[row + 2]; idx++) {
                t1 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 2 < this->rows)
            for (idx = this->getIRP()[row + 2]; idx < this->getIRP()[row + 3]; idx++) {
                t2 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 3 < this->rows)
            for (idx = this->getIRP()[row + 3]; idx < this->getIRP()[row + 4]; idx++) {
                t3 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        y[row] = t0;
        y[row + 1] = t1;
        y[row + 2] = t2;
        y[row + 3] = t3;
    }

    return y;
}

double *MatrixCSR::openMPMultiplyUnroll8V(double *x, double *y) {

    if (x == nullptr || y == nullptr) {
        std::cerr << "Error: x or y pointer is null" << std::endl;
        return nullptr;
    }

    if (this->cols != this->rows) {
        std::cerr << "Error: Matrix dimensions are not valid" << std::endl;
        return nullptr;
    }

    int row,  idx;
#pragma omp parallel for shared(x, y) private(row,  idx )
    for (row = 0; row < rows - rows % 8; row += 8) {
        double t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
        for (idx = this->getIRP()[row]; idx < this->getIRP()[row + 1]; idx++) {
            t0 += this->getAS()[idx] * x[this->getJA()[idx]];
        }
        if (row + 1 < this->rows)
            for (idx = this->getIRP()[row + 1]; idx < this->getIRP()[row + 2]; idx++) {
                t1 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 2 < this->rows)
            for (idx = this->getIRP()[row + 2]; idx < this->getIRP()[row + 3]; idx++) {
                t2 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 3 < this->rows)
            for (idx = this->getIRP()[row + 3]; idx < this->getIRP()[row + 4]; idx++) {
                t3 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 4 < this->rows)
            for (idx = this->getIRP()[row + 4]; idx < this->getIRP()[row + 5]; idx++) {
                t4 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 5 < this->rows)
            for (idx = this->getIRP()[row + 5]; idx < this->getIRP()[row + 6]; idx++) {
                t5 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 6 < this->rows)
            for (idx = this->getIRP()[row + 6]; idx < this->getIRP()[row + 7]; idx++) {
                t6 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 7 < this->rows)
            for (idx = this->getIRP()[row + 7]; idx < this->getIRP()[row + 8]; idx++) {
                t7 += this->getAS()[idx] * x[this->getJA()[idx]];
            }

        y[row] = t0;
        y[row + 1] = t1;
        y[row + 2] = t2;
        y[row + 3] = t3;
        y[row + 4] = t4;
        y[row + 5] = t5;
        y[row + 6] = t6;
        y[row + 7] = t7;
    }

    return y;
}

double *MatrixCSR::openMPMultiplyUnroll16V(double *x, double *y) {

    if (x == nullptr || y == nullptr) {
        std::cerr << "Error: x or y pointer is null" << std::endl;
        return nullptr;
    }

    if (this->cols != this->rows) {
        std::cerr << "Error: Matrix dimensions are not valid" << std::endl;
        return nullptr;
    }

    int row,  idx;

#pragma omp parallel for private(row,  idx) shared(y, x)
    for (row = 0; row < this->rows - rows % 8; row += 16) {
        double t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0, t14 = 0, t15 = 0;
        for (idx = this->getIRP()[row]; idx < this->getIRP()[row + 1]; idx++) {
            t0 += this->getAS()[idx] * x[this->getJA()[idx]];
        }
        if (row + 1 < this->rows)
            for (idx = this->getIRP()[row + 1]; idx < this->getIRP()[row + 2]; idx++) {
                t1 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 2 < this->rows)
            for (idx = this->getIRP()[row + 2]; idx < this->getIRP()[row + 3]; idx++) {
                t2 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 3 < this->rows)
            for (idx = this->getIRP()[row + 3]; idx < this->getIRP()[row + 4]; idx++) {
                t3 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 4 < this->rows)
            for (idx = this->getIRP()[row + 4]; idx < this->getIRP()[row + 5]; idx++) {
                t4 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 5 < this->rows)
            for (idx = this->getIRP()[row + 5]; idx < this->getIRP()[row + 6]; idx++) {
                t5 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 6 < this->rows)
            for (idx = this->getIRP()[row + 6]; idx < this->getIRP()[row + 7]; idx++) {
                t6 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 7 < this->rows)
            for (idx = this->getIRP()[row + 7]; idx < this->getIRP()[row + 8]; idx++) {
                t7 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 8 < this->rows)
            for (idx = this->getIRP()[row + 8]; idx < this->getIRP()[row + 9]; idx++) {
                t8 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 9 < this->rows)
            for (idx = this->getIRP()[row + 9]; idx < this->getIRP()[row + 10]; idx++) {
                t9 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 10 < this->rows)
            for (idx = this->getIRP()[row + 10]; idx < this->getIRP()[row + 11]; idx++) {
                t10 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 11 < this->rows)
            for (idx = this->getIRP()[row + 11]; idx < this->getIRP()[row + 12]; idx++) {
                t11 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 12 < this->rows)
            for (idx = this->getIRP()[row + 12]; idx < this->getIRP()[row + 13]; idx++) {
                t12 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 13 < this->rows)
            for (idx = this->getIRP()[row + 13]; idx < this->getIRP()[row + 14]; idx++) {
                t13 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 14 < this->rows)
            for (idx = this->getIRP()[row + 14]; idx < this->getIRP()[row + 15]; idx++) {
                t14 += this->getAS()[idx] * x[this->getJA()[idx]];
            }
        if (row + 15 < this->rows)
            for (idx = this->getIRP()[row + 15]; idx < this->getIRP()[row + 16]; idx++) {
                t15 += this->getAS()[idx] * x[this->getJA()[idx]];
            }

        y[row] = t0;
        y[row + 1] = t1;
        y[row + 2] = t2;
        y[row + 3] = t3;
        y[row + 4] = t4;
        y[row + 5] = t5;
        y[row + 6] = t6;
        y[row + 7] = t7;
        y[row + 8] = t8;
        y[row + 9] = t9;
        y[row + 10] = t10;
        y[row + 11] = t11;
        y[row + 12] = t12;
        y[row + 13] = t13;
        y[row + 14] = t14;
        y[row + 15] = t15;
    }

    return y;
}

double *MatrixCSR::openMPMultiplyUnroll2H(double *x, double *y) {
    if (x == nullptr || y == nullptr) {
        std::cerr << "Error: x or y pointer is null" << std::endl;
        return nullptr;
    }

    if (this->cols != this->rows) {
        std::cerr << "Error: Matrix dimensions are not valid" << std::endl;
        return nullptr;
    }

    int row, idx;

#pragma omp parallel for private(row,  idx) shared(y, x)
    for (row = 0; row < this->rows; row++) {
        double t0 = 0, t1 = 0;
        for (idx = this->getIRP()[row]; idx < this->getIRP()[row + 1]; idx += 2) {
            if (idx + 0 < this->getIRP()[row + 0])
                t0 += this->getAS()[idx] * x[this->getJA()[idx]];
            if (idx + 1 < this->getIRP()[row + 1])
                t1 += this->getAS()[idx + 1] * x[this->getJA()[idx + 1]];
        }

        y[row] = t0 + t1;
    }

#pragma omp barrier

    return y;
}

double *MatrixCSR::openMPMultiplyUnroll4H(double *x, double *y) {

    if (x == nullptr || y == nullptr) {
        std::cerr << "Error: x or y pointer is null" << std::endl;
        return nullptr;
    }

    if (this->cols != this->rows) {
        std::cerr << "Error: Matrix dimensions are not valid" << std::endl;
        return nullptr;
    }

    int row,  idx;

#pragma omp parallel for private(row,  idx) shared(y, x)
    for (row = 0; row < this->rows; row++) {
        double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (idx = this->getIRP()[row]; idx < this->getIRP()[row + 1]; idx += 4) {
            t0 += this->getAS()[idx] * x[this->getJA()[idx]];
            if (idx + 1 < this->getIRP()[row + 1])
            t1 += this->getAS()[idx + 1] * x[this->getJA()[idx + 1]];
            if (idx + 2 < this->getIRP()[row + 1])
            t2 += this->getAS()[idx + 2] * x[this->getJA()[idx + 2]];
            if (idx + 3 < this->getIRP()[row + 1])
            t3 += this->getAS()[idx + 3] * x[this->getJA()[idx + 3]];
        }
        y[row] = t0 + t1 + t2 + t3;
    }

#pragma omp barrier

    return y;
}

double *MatrixCSR::openMPMultiplyUnroll8H(double *x, double *y) {
    if (x == nullptr || y == nullptr) {
        std::cerr << "Error: x or y pointer is null" << std::endl;
        return nullptr;
    }

    if (this->cols != this->rows) {
        std::cerr << "Error: Matrix dimensions are not valid" << std::endl;
        return nullptr;
    }

    int row, idx;

#pragma omp parallel for private(row, idx) shared(y, x)
    for (row = 0; row < this->rows; row++) {
        double t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
        for (idx = this->getIRP()[row]; idx < this->getIRP()[row + 1]; idx += 8) {
            t0 += this->getAS()[idx] * x[this->getJA()[idx]];
            if (idx + 1 < this->getIRP()[row + 1])
                t1 += this->getAS()[idx + 1] * x[this->getJA()[idx + 1]];
            if (idx + 2 < this->getIRP()[row + 1])
                t2 += this->getAS()[idx + 2] * x[this->getJA()[idx + 2]];
            if (idx + 3 < this->getIRP()[row + 1])
                t3 += this->getAS()[idx + 3] * x[this->getJA()[idx + 3]];
            if (idx + 4 < this->getIRP()[row + 1])
                t4 += this->getAS()[idx + 4] * x[this->getJA()[idx + 4]];
            if (idx + 5 < this->getIRP()[row + 1])
                t5 += this->getAS()[idx + 5] * x[this->getJA()[idx + 5]];
            if (idx + 6 < this->getIRP()[row + 1])
                t6 += this->getAS()[idx + 6] * x[this->getJA()[idx + 6]];
            if (idx + 7 < this->getIRP()[row + 1])
                t7 += this->getAS()[idx + 7] * x[this->getJA()[idx + 7]];
        }
        y[row] = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7;
    }

#pragma omp barrier

    return y;
}


double *MatrixCSR::openMPMultiplyUnroll16H(double *x, double *y) {
    if (x == nullptr || y == nullptr) {
        std::cerr << "Error: x or y pointer is null" << std::endl;
        return nullptr;
    }

    if (this->cols != this->rows) {
        std::cerr << "Error: Matrix dimensions are not valid" << std::endl;
        return nullptr;
    }

    int row,  idx;

#pragma omp parallel for private(row,  idx) shared(y, x)
    for (row = 0; row < this->rows; row++) {
        double t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0, t14 = 0, t15 = 0;
        for (idx = this->getIRP()[row]; idx < this->getIRP()[row + 1]; idx += 16) {
            t0 += this->getAS()[idx] * x[this->getJA()[idx]];
            if (idx + 1 < this->getIRP()[row + 1])
                t1 += this->getAS()[idx + 1] * x[this->getJA()[idx + 1]];
            if (idx + 2 < this->getIRP()[row + 1])
                t2 += this->getAS()[idx + 2] * x[this->getJA()[idx + 2]];
            if (idx + 3 < this->getIRP()[row + 1])
                t3 += this->getAS()[idx + 3] * x[this->getJA()[idx + 3]];
            if (idx + 4 < this->getIRP()[row + 1])
                t4 += this->getAS()[idx + 4] * x[this->getJA()[idx + 4]];
            if (idx + 5 < this->getIRP()[row + 1])
                t5 += this->getAS()[idx + 5] * x[this->getJA()[idx + 5]];
            if (idx + 6 < this->getIRP()[row + 1])
                t6 += this->getAS()[idx + 6] * x[this->getJA()[idx + 6]];
            if (idx + 7 < this->getIRP()[row + 1])
                t7 += this->getAS()[idx + 7] * x[this->getJA()[idx + 7]];
            if (idx + 8 < this->getIRP()[row + 1])
                t8 += this->getAS()[idx + 8] * x[this->getJA()[idx + 8]];
            if (idx + 9 < this->getIRP()[row + 1])
                t9 += this->getAS()[idx + 9] * x[this->getJA()[idx + 9]];
            if (idx + 10 < this->getIRP()[row + 1])
                t10 += this->getAS()[idx + 10] * x[this->getJA()[idx + 10]];
            if (idx + 11 < this->getIRP()[row + 1])
                t11 += this->getAS()[idx + 11] * x[this->getJA()[idx + 11]];
            if (idx + 12 < this->getIRP()[row + 1])
                t12 += this->getAS()[idx + 12] * x[this->getJA()[idx + 12]];
            if (idx + 13 < this->getIRP()[row + 1])
                t13 += this->getAS()[idx + 13] * x[this->getJA()[idx + 13]];
            if (idx + 14 < this->getIRP()[row + 1])
                t14 += this->getAS()[idx + 14] * x[this->getJA()[idx + 14]];
            if (idx + 15 < this->getIRP()[row + 1])
                t15 += this->getAS()[idx + 15] * x[this->getJA()[idx + 15]];
        }

        y[row] = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t14 + t15;
    }

#pragma omp barrier

    return y;
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

void MatrixCSR::setIRP(int nz, int *I) {
    // Initialize the row pointer for the first row.
    IRP[0] = 0;

    int row = 0;
    int idx = 0;

    // Iterate through all the non-zero elements of the I array.
    for (int i = 0; i < nz; i++) {
        // If the row of the current element is different than the previous element,
        // then update the row pointer for the new row and move to the next row.
        if (I[i] > row) {
            while (row < I[i]) {
                IRP[++row] = idx;
            }
        }
        idx++;
    }

    // If there are any remaining rows with no non-zero elements, then update their row pointers.
    while (row < nz - 1) {
        IRP[++row] = nz;
    }
}


void MatrixCSR::setJA(int nz, int *I, int *J) {
    //call the base function
    for (int i = 0; i < nz; i++) {
        this->JA[i] = J[i];
    }
}

void MatrixCSR::setAS(int nz, double *val) {
    //call the base function
    for (int i = 0; i < nz; i++) {
        this->AS[i] = val[i];
    }
}
