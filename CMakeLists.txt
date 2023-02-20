#####LOCAL SYSTEM CONFIGURATION TO RUN WITH IDE#####
cmake_minimum_required(VERSION 3.15.3)
project(SmallScale)
message(STATUS "set compiler")
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
set(OPENMP_LIBRARIES "/opt/homebrew/opt/llvm/lib")
set(OPENMP_INCLUDES "/opt/homebrew/opt/llvm/include")
set(OPENMP_LIBRARIES "/opt/homebrew/opt/llvm/lib")
link_directories(${OPENMP_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O4")

#add_subdirectory(CUDA)
#add_subdirectory(src)
add_executable(SmallScale src/MatrixBase.cpp src/main.cpp src/mmio.cpp src/OMP/MatrixCSR.cpp src/OMP/MatrixELLPACK.cpp src/wtime.cpp)
