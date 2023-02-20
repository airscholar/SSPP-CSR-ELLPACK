cuda_add_executable(CSR_1T MatrixCSR_1thread.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)

#1D Thread
cuda_add_executable(CSR_1DB MatrixCSR_1DB.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(CSR_1DB_2H MatrixCSR_1DB_2H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(CSR_1DB_4H MatrixCSR_1DB_4H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(CSR_1DB_8H MatrixCSR_1DB_8H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(CSR_1DB_16H MatrixCSR_1DB_16H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)

#2D Block
cuda_add_executable(CSR_2DB MatrixCSR_2DB.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(CSR_2DB_2H MatrixCSR_2DB_2H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(CSR_2DB_4H MatrixCSR_2DB_4H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(CSR_2DB_8H MatrixCSR_2DB_8H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(CSR_2DB_16H MatrixCSR_2DB_16H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)


