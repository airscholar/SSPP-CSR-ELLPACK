#include <cstdio>
#include <cstdlib>
#include "mmio.h"
//#include "smallscale.h"
#include <utility>
#include "MatrixBase.h"
#include "OMP/MatrixCSR.h"
#include "OMP/MatrixELLPACK.h"
#include "wtime.h"
#include <map>
#include <omp.h>
#include <string>

using namespace std;

inline double dmin ( double a, double b ) { return a < b ? a : b; }
map<pair<int, int>, double> matrix; // use a map to store the values of I, J and V
const int ntimes = 20;

//void readFile(int &M, int &N, int &nz, int *&I, int *&J, double *&val, int &ret_code, MM_typecode &matcode, char *fileName) {
//    FILE *f;
//    if ((f = fopen(fileName, "r")) == NULL)
//        exit(1);
//
//    if (mm_read_banner(f, &matcode) != 0) {
//        printf("Could not process Matrix Market banner.\n");
//        exit(1);
//    }
//
//    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
//        mm_is_sparse(matcode)) {
//        printf("Sorry, this application does not support ");
//        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
//        exit(1);
//    }
//
//    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
//        exit(1);
//
//    printf("MATCODE => %s\n", mm_typecode_to_str(matcode));
//
//    // Reserve memory for the input matrix
//    I = new int[nz];
//    J = new int[nz];
//    val = new double[nz];
//
//    // Read the matrix data into I, J, val
//    mm_read_mtx_crd_data(f, M, N, nz, I, J, val, matcode);
//
//    // Allocate temporary arrays to hold the input data
//    int *tempI = new int[nz];
//    int *tempJ = new int[nz];
//    double *tempVal = new double[nz];
//
//    // Copy the input data into the temporary arrays
//    for (int i = 0; i < nz; i++) {
//        tempI[i] = I[i];
//        tempJ[i] = J[i];
//        tempVal[i] = val[i];
//    }
//
//    // Compute the number of diagonal elements
//    int numDiag = 0;
//    for (int i = 0; i < nz; i++) {
//        if (tempI[i] == tempJ[i]) {
//            numDiag++;
//        }
//    }
//
//    // If the matrix is symmetric, adjust the number of non-zero elements
//    if (mm_is_symmetric(matcode)) {
//        // Count the number of off-diagonal elements
//        int numOffDiag = nz - numDiag;
//
//        // Allocate memory for the new arrays
//        int newNz = nz + numOffDiag;
//        int *newI = new int[newNz];
//        int *newJ = new int[newNz];
//        double *newVal = new double[newNz];
//
//        // Copy the original matrix data into the new arrays, making it symmetric
//        int k = 0;
//        for (int i = 0; i < nz; i++) {
//            newI[k] = tempI[i];
//            newJ[k] = tempJ[i];
//            newVal[k] = tempVal[i];
//            k++;
//            if (tempI[i] != tempJ[i]) {
//                newI[k] = tempJ[i];
//                newJ[k] = tempI[i];
//                newVal[k] = tempVal[i];
//                k++;
//            }
//        }
//
//        // Free the old arrays and update the pointers to point to the new arrays
//        delete[] I;
//        delete[] J;
//        delete[] val;
//        I = newI;
//        J = newJ;
//        val = newVal;
//    }
//}

void readFile(int &M, int &N, int &nz, int *&I, int *&J, double *&val, int &ret_code, MM_typecode &matcode, char *fileName) {
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

    // Get the size of the sparse matrix
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
        printf("Error: could not read matrix size.\n");
        exit(1);
    }

    // Allocate memory for the matrices
    I = new int[nz];
    J = new int[nz];
    val = new double[nz];

    // Read the data
    mm_read_mtx_crd_data(f, M, N, nz, I, J, val, matcode);

    // Convert the matrix to a symmetric format (if needed)
    int diagonal = 0;
    if (mm_is_symmetric(matcode)) {
        for (int i = 0; i < nz; i++) {
            if (I[i] == J[i]) {
                diagonal++;
            }
        }
        int oldNz = nz;
        nz = nz * 2 - diagonal;
        I = (int *) realloc(I, nz * sizeof(int));
        J = (int *) realloc(J, nz * sizeof(int));
        val = (double *) realloc(val, nz * sizeof(double));
        int k = oldNz;
        for (int i = 0; i < oldNz; i++) {
            if (I[i] != J[i]) {
                I[k] = J[i];
                J[k] = I[i];
                val[k] = val[i];
                k++;
            }
        }
    }

    // Create a map to store the matrix
    for (int i = 0; i < nz; i++) {
        matrix[make_pair(I[i], J[i])] = val[i];
    }

    // Close the file
    fclose(f);
}

void printMatrix(int M, int N){
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

double* generateVector(int rows){
    double* A = new double[rows];

    int row;

//    srand(12345);
    for (row = 0; row < rows; row++) {
//        A[row] = 100.0f * ((double) rand()) / RAND_MAX;
        A[row] = 1;
    }

    return A;
}

void CSRResult(char *fileName, double *x, double *y, int M, int N, int nz, int *I, int *J, double *val, MatrixCSR matrixCSR) {
    // multiply
    double tmlt = 1e100;
    double *serialCSRResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        serialCSRResult = matrixCSR.serialMultiply(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }

    double serialGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u2VResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u2VResult = matrixCSR.openMPMultiplyUnroll2V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }

    double u2VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u2HResult = 0;
    for (int tr = 0; tr < M; tr++) {
        double t1 = wtime();
        u2HResult = matrixCSR.openMPMultiplyUnroll2H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }

    double u2HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u4HResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u4HResult = matrixCSR.openMPMultiplyUnroll4H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }

    double u4HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u4VResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u4VResult = matrixCSR.openMPMultiplyUnroll4V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }

    double u4VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u8VResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u8VResult = matrixCSR.openMPMultiplyUnroll8V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    double u8VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u8HResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u8HResult = matrixCSR.openMPMultiplyUnroll8H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    double u8HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u16HResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u16HResult = matrixCSR.openMPMultiplyUnroll16H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    double u16HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u16VResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u16VResult = matrixCSR.openMPMultiplyUnroll16V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    double u16VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    ;

    //display the results
    //headers
    printf("DataSet\t\t\t\t\tNumProc\tSerial\t\tu2v\t\tu4v\t\tu8v\t\tu16v\t\tu2h\t\tu4h\t\tu8h\t\tu16h\t\n");
#pragma omp parallel
    {
#pragma omp master
        {
    printf("%s\t\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t \n", fileName, omp_get_num_threads(), serialGflops, u2VGflops, u4VGflops, u8VGflops, u16VGflops, u2HGflops, u4HGflops, u8HGflops, u16HGflops);
        }
    };
}

void ELLPACKResult(char *fileName, double *x, double *y, int M, int N, int nz, int *I, int *J, double *val, MatrixELLPACK matrixELLPack) {
    // multiply
    double tmlt = 1e100;
    double *serialCSRResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        serialCSRResult = matrixELLPack.serialMultiply(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }

    double serialGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u2VResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u2VResult = matrixELLPack.openMPMultiplyUnroll2V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }

    double u2VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u2HResult = 0;
    for (int tr = 0; tr < M; tr++) {
        double t1 = wtime();
        u2HResult = matrixELLPack.openMPMultiplyUnroll2H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }

    double u2HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u4HResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u4HResult = matrixELLPack.openMPMultiplyUnroll4H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }

    double u4HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u4VResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u4VResult = matrixELLPack.openMPMultiplyUnroll4V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }

    double u4VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u8VResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u8VResult = matrixELLPack.openMPMultiplyUnroll8V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    double u8VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u8HResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u8HResult = matrixELLPack.openMPMultiplyUnroll8H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    double u8HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u16HResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u16HResult = matrixELLPack.openMPMultiplyUnroll16H(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    double u16HGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    tmlt = 1e100;
    double *u16VResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        u16VResult = matrixELLPack.openMPMultiplyUnroll16V(x, y);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    double u16VGflops = (2.0 * nz / tmlt * 1e-6) * 0.001;

    ;

    //display the results
    //headers
    printf("DataSet\t\t\t\t\tNumProc\tSerial\t\tu2v\t\tu4v\t\tu8v\t\tu16v\t\tu2h\t\tu4h\t\tu8h\t\tu16h\t\n");
#pragma omp parallel
    {
#pragma omp master
        {
            printf("%s\t\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t \n", fileName, omp_get_num_threads(), serialGflops, u2VGflops, u4VGflops, u8VGflops, u16VGflops, u2HGflops, u4HGflops, u8HGflops, u16HGflops);
        }
    };
}

int main(int argc, char *argv[]) {
    int ret_code;
    MM_typecode matcode;

    int M, N, nz;
    int *I, *J;
    double *val;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    }

    readFile(M, N, nz, I, J, val, ret_code, matcode, argv[1]);


    MatrixBase::sortData(I, J, val, nz);

    //generate X vector
    double *x = generateVector(M);
    //get start time
    double *y = new double[M];

    MatrixCSR matrixCSR(M, N, nz, I, J, val, x);
    CSRResult(argv[1], x, y, M, N, nz, I, J, val, matrixCSR);

// multiply
//    double tmlt = 1e100;
//    double *serialCSRResult = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        double t1 = wtime();
//        serialCSRResult = matrixCSR.serialMultiply(x, y);
//        double t2 = wtime();
//        tmlt = dmin(tmlt, (t2 - t1));
//    }

    //validate result
//    float diff = 0;
//    for (int i = 0; i < M; i++) {
//        float err = serialCSRResult[i] - u8HResult[i];
//        if (err < 0) err = -err;
//        printf("serial = %f \t u8H = %f \t err = %f \n", serialCSRResult[i], u8HResult[i], err);
//        diff += err;
//    }
//    printf("Error: %f \n", diff);

    printf("ELLPACK");
    MatrixELLPACK ellpack(M, N, nz, I, J, val, x);

    ELLPACKResult(argv[1], x, y, M, N, nz, I, J, val, ellpack);
//
//    double* y1 = new double[M];
//    double tmlt1 = 1e100;
//    double *serialEllPackResult = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        long double t1 = wtime();
//        serialEllPackResult = ellpack.serialOMPMultiply(x, y1);
//        double t2 = wtime();
//        tmlt1 = dmin(tmlt1, (t2 - t1));
//    }
//    double gflops1 = (2.0 * nz / tmlt1 * 1e-6) * 0.001;
//    printf("Serial ELLPACK %d x %d: time %lf  GFLOPS: %f \n", M, N, tmlt1, gflops1);
//
//    tmlt1 = 1e100;
//    double *u2Hresult = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        double t1 = wtime();
//        u2Hresult = ellpack.openMPMultiplyUnroll16V(x, y);
//        double t2 = wtime();
//        tmlt1 = dmin(tmlt1, (t2 - t1));
//    }
//    gflops1 = (2.0 * nz / tmlt1 * 1e-6) * 0.001;
//    printf("OMP ELLPACK %d x %d: time %lf  GFLOPS: %f \n", M, N, tmlt1, gflops1);
//
//    //validate result
//    double diff1 = 0;
//    for (int i = 0; i < M; i++) {
//        float err = serialEllPackResult[i] - u2Hresult[i];
//        if (err < 0) err = -err;
//        if(err != 0) printf("serial = %f \t u2H = %f \t err = %f \n", serialEllPackResult[i], u2Hresult[i], err);
//        diff1 += err;
//    }
//    printf("Error: %f \n", diff1);
//
//
//    tmlt1 = 1e100;
//    double *ompEllpackH = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        double t1 = wtime();
//        ompEllpackH = ellpack.openMPMultiplyUnroll2Hor(x, y);
//        double t2 = wtime();
//        tmlt1 = dmin(tmlt1, (t2 - t1));
//    }
//    gflops1 = (2.0 * nz / tmlt1 * 1e-6) * 0.001;
//    printf("2UHR ELLPACK %d x %d: time %lf  GFLOPS: %f \n", M, N, tmlt1, gflops1);
//    //validate result
//     diff1 = 0;
//    for (int i = 0; i < M; i++) {
//        float err = ompELLPackresult[i] - ompEllpackH[i];
//        if (err < 0) err = -err;
//        diff1 += err;
//    }
//    printf("Error: %f \n", diff1);
//    tmlt1 = 1e100;
//    double *unroll2HVresult = 0;
//    for (int tr = 0; tr < ntimes; tr++) {
//        double t1 = wtime();
//        unroll2HVresult = ellpack.openMPMultiplyUnroll2HorVert(x, y);
//        double t2 = wtime();
//        tmlt1 = dmin(tmlt1, (t2 - t1));
//    }
//    gflops1 = (2.0 * nz / tmlt1 * 1e-6) * 0.001;
//    printf("UHV time %lf  GFLOPS: %f \n", M, N, tmlt1, gflops1);

    delete[] x;
    delete[] y;
    delete[] I;
    delete[] J;
    delete[] val;

    return 0;
}