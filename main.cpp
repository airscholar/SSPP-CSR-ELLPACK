#include <cstdio>
#include <cstdlib>
#include "mmio.h"
#include "smallscale.h"
#include <vector>
#include <utility>
#include "MatrixCSR.h"
#include "MatrixELLPACK.h"
#include "wtime.h"

using namespace std;

int main(int argc, char *argv[]) {
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int i, *I, *J;
    double *val;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
        exit(1);
    } else {
        if ((f = fopen(argv[1], "r")) == NULL)
            exit(1);
    }

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


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int)); // row
    J = (int *) malloc(nz * sizeof(int)); // col
    val = (double *) malloc(nz * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < nz; i++) {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    if (f != stdin) fclose(f);

    printf("Dimensions: %d x %d, Non-zero elements: %d\n", M, N, nz);

    //get start time
    double t1 = wtime();
    MatrixCSR csr(M, N, nz, I, J, val);
    printf("=====================================\n");
    double t2 = wtime();
    printf("Time: %f\n", t2 - t1);
    double time = t2 - t1;
    double mflops = (2.0e-6) * M * N / time;
    fprintf(stdout,"Multiplying matrices of size %d x %d: time %lf  MFLOPS %lf \n",M,N,time,mflops);


//    printf("ELLPACK\n");
//    MatrixELLPACK ellpack(M, N, nz, I, J, val);

    //free memory
    free(I);
    free(J);
    free(val);

    return 0;
}