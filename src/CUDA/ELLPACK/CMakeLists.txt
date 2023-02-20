#ELLPACK
cuda_add_executable(EBlock1TH ELLPACK_1thread.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)

cuda_add_executable(EBlock1DB ELLPACK_1DB.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(EBlock1DB_2H ELLPACK_1DB_2H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(EBlock1DB_4H ELLPACK_1DB_4H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(EBlock1DB_8H ELLPACK_1DB_8H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(EBlock1DB_16H ELLPACK_1DB_16H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)

cuda_add_executable(EBlock2DB ELLPACK_2DB.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(EBlock2DB_2H ELLPACK_2DB_2H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(EBlock2DB_4H ELLPACK_2DB_4H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(EBlock2DB_8H ELLPACK_2DB_8H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)
cuda_add_executable(EBlock2DB_16H ELLPACK_2DB_16H.cu ../../MatrixBase.cpp ../../OMP/MatrixCSR.cpp ../../OMP/MatrixELLPACK.cpp ../../wtime.cpp ../../mmio.cpp)

