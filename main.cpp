#include <cstdio>
#include <cstdlib>
#include "mmio.h"
#include <iostream>
#include <string>
#include <vector>
#include <utility>

using namespace std;

vector<vector<int>> convertInputToMatrix(int rows, int cols, int nonZeroElements, int *I, int *J, double *val) {
    vector<vector<int>> matrix(rows);

    //initialize the matrix
    for (int i = 0; i < rows; i++) {
        matrix[i].resize(cols);
    }

    //fill the matrix
    for (int i = 0; i < nonZeroElements; i++) {
        matrix[I[i]][J[i]] = val[i];
    }

    return matrix;
}

vector<int> extractIRP(vector<vector<int>> input){
    vector<int> irp;
    int count = 0;
    for (int i = 0; i < input.size(); i++) {
        int row_count = 0;
        for (int j = 0; j < input[i].size(); j++) {
            if(input[i][j] != 0) {
                row_count++;
                count++;
                if(row_count == 1) {
                    irp.push_back(count);
                }
            }
        }
        if(i == input.size()-1) {
            irp.push_back(count+1);
        }
    }

    return irp;
}

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

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
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

//    mm_write_banner(stdout, matcode);
//    mm_write_mtx_crd_size(stdout, M, N, nz);

//    for (i = 0; i < nz; i++)
//        fprintf(stdout, "%d %d %20.19g\n", I[i] + 1, J[i] + 1, val[i]);

    vector<vector<int>> input = convertInputToMatrix(M, N, nz, I, J, val);

    //the amtrix
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < input[i].size(); j++) {
            printf("%d ", input[i][j]);
        }
        printf("\n");
    }

    vector<int> irp = extractIRP(input);
    printf("IRP: ");
    for (int i = 0; i < irp.size(); i++) {
        printf("%d ", irp[i]);
    }
    printf("\n");


    //print the column index
    printf("JA: ");
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < input[i].size(); j++) {
            if (input[i][j] != 0) {
                printf("%d ", j+1);
            }
        }
    }
    printf("\n");

    //print the non-zero elements
    printf("AS: ");
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < input[i].size(); j++) {
            if (input[i][j] != 0) {
                printf("%d ", input[i][j]);
            }
        }
    }


    //free memory
    free(I);
    free(J);
    free(val);

    return 0;
}
