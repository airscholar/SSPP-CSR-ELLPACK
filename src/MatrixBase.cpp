//
// Created by Yusuf Ganiyu on 2/14/23.
//
#include <string>
#include <vector>
#include "MatrixBase.h"
#include "mmio.h"
#include "SparseMatrix.h"

double* MatrixBase::generateVector(int vectorSize) {
    double *vector = new double[vectorSize];
    for (int i = 0; i < vectorSize; i++) {
        vector[i] = 1;
    }
    return vector;
}

void MatrixBase::readFile(int &M, int &N, int &nz, int *&I, int *&J, double *&val, char *fileName) {
    int ret_code;
    MM_typecode matcode;

    // Open the file
    FILE *f = fopen(fileName, "r");
    if (f == NULL) {
        printf("Error: could not open file.\n");
        exit(1);
    }

    // Read the Matrix Market banner
    if (mm_read_banner(f, &matcode) != 0) {
        printf("Error: could not process Matrix Market banner.\n");
        exit(1);
    }

    // Check if the matrix type is supported
    if (mm_is_complex(matcode) || !mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
        printf("Error: unsupported matrix type [%s].\n", mm_typecode_to_str(matcode));
        exit(1);
    }

//    printf("Matrix type: %s \n", mm_typecode_to_str(matcode));
    // Get the size of the sparse matrix
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
        printf("Error: could not read matrix size.\n");
        exit(1);
    }

    // Allocate memory for the matrices
    int *tempI = new int[nz];
    int *tempJ = new int[nz];
    double *tempVal = new double[nz];

    // Read the data
    mm_read_mtx_crd_data(f, M, N, nz, tempI, tempJ, tempVal, matcode);

    //if the matrix is pattern, set all values to 1
    if (mm_is_pattern(matcode)) {
        for (int i = 0; i < nz; i++) {
            tempVal[i] = 1;
        }
    }

    // Convert the matrix to a symmetric format (if needed)
    int diagonal = 0;
    if (mm_is_symmetric(matcode)) {
        for (int i = 0; i < nz; i++) {
            if (tempI[i] == tempJ[i]) {
                diagonal++;
            }
        }
        int oldNz = nz;
        nz = nz * 2 - diagonal;
        tempI = (int *) realloc(tempI, nz * sizeof(int));
        tempJ = (int *) realloc(tempJ, nz * sizeof(int));
        tempVal = (double *) realloc(tempVal, nz * sizeof(double));
        int k = oldNz;
        for (int i = 0; i < oldNz; i++) {
            if (tempI[i] != tempJ[i]) {
                tempI[k] = tempJ[i];
                tempJ[k] = tempI[i];
                tempVal[k] = tempVal[i];
                k++;
            }
        }
    }

    // assign value to the pointers
    I = new int[nz];
    J = new int[nz];
    val = new double[nz];

    //assign values
    std::copy(tempI, tempI + nz, I);
    std::copy(tempJ, tempJ + nz, J);
    std::copy(tempVal, tempVal + nz, val);

    // Free the temporary memory
    delete[] tempI;
    delete[] tempJ;
    delete[] tempVal;

    // Close the file
    fclose(f);
}

void MatrixBase::sortData(int* &I_input, int* &J_input, double* &AS_input, int nz_input) {
    // create an index array
    std::vector<int> index(nz_input);
    for (int i = 0; i < nz_input; ++i) {
        index[i] = i;
    }

    // sort the index array based on the values in I_input and J_input
    std::sort(index.begin(), index.end(), [&](int a, int b) {
        return I_input[a] < I_input[b] ||
               (I_input[a] == I_input[b]) && J_input[a] < J_input[b];
    });

    // create temporary arrays to store the sorted values
    std::vector<int> J_sorted(nz_input);
    std::vector<int> I_sorted(nz_input);
    std::vector<double> AS_sorted(nz_input);

    // sort the J_input, AS_input and I_input arrays in parallel
    for (int i = 0; i < nz_input; ++i) {
        int j = index[i];
        I_sorted[i] = I_input[j]-1;
        J_sorted[i] = J_input[j]-1;
        AS_sorted[i] = AS_input[j];
    }

    //assign values
    std::copy(I_sorted.begin(), I_sorted.end(), I_input);
    std::copy(J_sorted.begin(), J_sorted.end(), J_input);
    std::copy(AS_sorted.begin(), AS_sorted.end(), AS_input);

    // Free the temporary memory
    index.clear();
    index.shrink_to_fit();
    J_sorted.clear();
    J_sorted.shrink_to_fit();
    AS_sorted.clear();
    AS_sorted.shrink_to_fit();
    I_sorted.clear();
    I_sorted.shrink_to_fit();
}

double MatrixBase::compute_Max_Error(double *x, double *y, int rows) {
    double maxError = 0;
    for (int i = 0; i < rows; i++) {
        double error = std::fabs(x[i] - y[i]);
        if (error > maxError) {
            maxError = error;
        }
    }
    return maxError;
}

void MatrixBase::printMatrix(int M, int N, bool showImage = false) {
    if (showImage == true) {
        int i, j;
        FILE *fp = fopen("matrix.ppm", "wb"); /* b - binary mode */
        (void) fprintf(fp, "P6\n%d %d\n255\n", M, N);
        for (i = 0; i < M; ++i) {
            for (j = 0; j < N; ++j) {
                static unsigned char color[3];
                color[0] = 0;  /* red */
                color[1] = 0;  /* green */
                color[2] = 0;  /* blue */
                if (matrix.count(make_pair(i, j)) || matrix.count(make_pair(j, i))) {
                    color[0] = 255;
                    color[1] = 255;
                    color[2] = 255;
                }
                (void) fwrite(color, 1, 3, fp);
            }
        }
    } else {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (matrix.count(make_pair(i, j)) || matrix.count(make_pair(j, i))) {
                    printf("0 ");
//                printf("%.2f\t", matrix[make_pair(i, j)]);
                } else {
                    printf("X ");
                }
            }
            printf("\n");
        }
    }
}

void MatrixBase::CSR_CpuMatrixVector(int rows, int *IRP, int *JA, double *AS, double *x, double *y) {
    for (int i = 0; i < rows; i++) {
        double sum = 0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            sum += AS[j] * x[JA[j]];
        }
        y[i] = sum;
    }
}

void MatrixBase::ELL_CpuMatrixVector(int rows, int *JA, double *AS, int maxNZ, double *x, double *y) {
    for (int i = 0; i < rows; i++) {
        double sum = 0;
        for (int j = 0; j < maxNZ; j++) {
            sum += AS[i * maxNZ + j] * x[JA[i * maxNZ + j]];
        }
        y[i] = sum;
    }
}