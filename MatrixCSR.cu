#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

int main(int argc, char** argv) {

    if (argc < 3) {
        fprintf(stderr, "Usage: %s  rows cols\n", argv[0]);
    }
    int nrows = atoi(argv[1]);
    int ncols = atoi(argv[2]);


}