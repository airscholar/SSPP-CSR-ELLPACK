#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O4")
#message(STATUS "set compiler")
#set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
#set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
#set(OPENMP_LIBRARIES "/opt/homebrew/opt/llvm/lib")
#set(OPENMP_INCLUDES "/opt/homebrew/opt/llvm/include")
#set(OPENMP_LIBRARIES "/opt/homebrew/opt/llvm/lib")
#link_directories(${OPENMP_LIBRARIES})
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O4")
add_subdirectory(CSR)
add_subdirectory(ELLPACK)


