//
// Created by Yusuf Ganiyu on 2/14/23.
//
#include <string>
#include <vector>
#include <unordered_map>
#include "MatrixBase.h"
#include <algorithm>

MatrixBase::MatrixBase(int rows, int cols, int nz, int *I, int *J, double *val, double *x) {
    this->rows = rows;
    this->cols = cols;
    this->nz = nz;
    this->I = I;
    this->J = J;
    this->val = val;
    this->x = x;
    this->y = new double[rows];
    this->IRP = new int[rows + 1];
    this->JA = new int[nz];
    this->AS = new double[nz];
}

MatrixBase::MatrixBase(int rows, int cols, int nz, int *I, int **JD, double *val, double *x) {
    this->rows = rows;
    this->cols = cols;
    this->nz = nz;
    this->I = I;
    this->JD = JD;
    this->val = val;
    this->x = x;
    this->y = new double[rows];
    this->IRP = new int[rows + 1];
    this->JA = new int[nz];
    this->AS = new double[nz];
}

void MatrixBase::sortData(int *I_, int *J_, double *AS_, int nz) {
// create an index array to keep track of the original indices
    std::vector<int> index(nz);
    for (int i = 0; i < nz; ++i) {
        index[i] = i;
    }

// sort the I_temp and index arrays in parallel
    std::sort(index.begin(), index.end(), [&](int a, int b) {
        return I_[a] < I_[b] ||
               (I_[a] == I_[b]) && J_[a] < J_[b];
    });

// create temporary arrays to hold the sorted J_temp and AS_temp
    std::vector<int> J_temp_sorted(nz);
    std::vector<double> AS_temp_sorted(nz);
    std::vector<int> I_temp_sorted(nz);

// recompose the J_temp and AS_temp arrays based on the sorted index array
    for (int i = 0; i < nz; ++i) {
        int j = index[i];
        I_temp_sorted[i] = I_[j]-1;
        J_temp_sorted[i] = J_[j]-1;
        AS_temp_sorted[i] = AS_[j];
    }

// overwrite the original J_temp and AS_temp arrays with the sorted versions
    std::copy(I_temp_sorted.begin(), I_temp_sorted.end(), I_);
    std::copy(J_temp_sorted.begin(), J_temp_sorted.end(), J_);
    std::copy(AS_temp_sorted.begin(), AS_temp_sorted.end(), AS_);

    //free memory
    index.clear();
    index.shrink_to_fit();
    J_temp_sorted.clear();
    J_temp_sorted.shrink_to_fit();
    AS_temp_sorted.clear();
    AS_temp_sorted.shrink_to_fit();
    I_temp_sorted.clear();
    I_temp_sorted.shrink_to_fit();
}

