project(gpgpu)
cmake_minimum_required(VERSION 2.8)

# Compute capability. See -arch option in nvcc --help.
set (CC 35)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

set(CMAKE_CXX_FLAGS "-O4")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)

add_subdirectory(CSR)
add_subdirectory(ELLPACK)