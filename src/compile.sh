g++ -std=c++11 -fopenmp -O4 MatrixBase.cpp main.cpp mmio.cpp OMP/MatrixCSR.cpp OMP/MatrixELLPACK.cpp wtime.cpp OMP_Tests/MatrixCSR_Test.cpp -o main 

