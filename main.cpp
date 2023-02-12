#include <cstdio>
#include <cstdlib>
#include "mmio.h"
#include "smallscale.h"
#include <utility>
#include "MatrixCSR.h"
#include "MatrixELLPACK.h"
#include "wtime.h"
#include <map>
#include <omp.h>

using namespace std;

inline double dmin ( double a, double b ) { return a < b ? a : b; }
map<pair<int, int>, double> matrix; // use a map to store the values of I, J and V
const int ntimes = 5;

void readFile(int &M, int &N, int &nz, int *&I, int *&J, double *&val, int &ret_code, MM_typecode &matcode, char *fileName) {
    FILE *f;
    if ((f = fopen(fileName, "r")) == NULL)
        exit(1);

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
        mm_is_sparse(matcode)) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }
    /* find out size of sparse matrix .... */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
        exit(1);

    printf("MATCODE => %s\n", matcode);

    /* reseve memory for matrices */

    if (mm_is_symmetric(matcode)) {
        nz = nz * 2;
    }
    I = (int *) malloc(nz * sizeof(int)); // row
    J = (int *) malloc(nz * sizeof(int)); // col
    val = (double *) malloc(nz * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    int i;
    char buffer[64];
    for (i = 0; i < nz; i++) {
        if (fgets(buffer, 64, f) == NULL) {
            break;
        }
        sscanf(buffer, "%d %d %lg", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;

        matrix[make_pair(I[i], J[i])] = val[i];

        // if symmetric, add the other half when its not the diagonal
        if (mm_is_symmetric(matcode) && I[i] != J[i]) {
            matrix[make_pair(J[i], I[i])] = val[i];
        }
    }

    if (f != stdin) fclose(f);
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
    double* A = (double*) malloc(sizeof(double)*rows);

    int row;

    srand(12345);
    for (row = 0; row < rows; row++) {
        A[row] = 100.0f * ((double) rand()) / RAND_MAX;
    }

    return A;
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

    printf("Dimensions: %d x %d, Non-zero elements: %d\n", M, N, nz);

//    printMatrix(M, N);

    //generate X vector
    double *x = generateVector(M);

    //get start time
    printf("=====================================\n");
    MatrixCSR csr(M, N, nz, I, J, val, x);
    // multiply
    double tmlt = 1e100;
    double *ompCSRresult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        ompCSRresult = csr.openMPMultiply(x);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    double mflops = (2.0e-6) * nz / tmlt;
    printf("CSR %d x %d: time %lf  MFLOPS: %f \n", M, N, tmlt, mflops);
//    //print csr result
//    printf("CSR Result:\t\t");
//    for (int i = 0; i < M; i++) {
//        printf("%.2f ", ompCSRresult[i]);
//    }
//    printf("\n");

    MatrixELLPACK ellpack(M, N, nz, I, J, val, x);

    tmlt = 1e100;
    double *ompEllpackResult = 0;
    for (int tr = 0; tr < ntimes; tr++) {
        double t1 = wtime();
        ompEllpackResult = ellpack.OMPMultiplyELLPack(x);
        double t2 = wtime();
        tmlt = dmin(tmlt, (t2 - t1));
    }
    mflops = (2.0e-6) * nz / tmlt;
    printf("ELLPACK %d x %d: time %lf  MFLOPS: %f \n", M, N, tmlt, mflops);
//    //print ELLPACK result
//    printf("ELLPACK Result:\t");
//    for (int i = 0; i < M; i++) {
//        printf("%.2f ", ompEllpackResult[i]);
//    }
//    printf("\n");

    //find the difference
    float diff = 0;
    for (int i = 0; i < M; i++) {
        diff += abs((float) ompCSRresult[i] - (float) ompEllpackResult[i]);
    }

    printf("Validation:\t\t%.9f\n", diff);
#pragma omp parallel
    {
#pragma omp master
        {
            printf("Computed results using %d threads\n", omp_get_num_threads());
        };
    }
    printf("=====================================\n");

    //free memory
    free(I);
    free(J);
    free(val);

    return 0;
}