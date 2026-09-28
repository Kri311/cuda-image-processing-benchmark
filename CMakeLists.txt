cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(CUDAImageBenchmark LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "native")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(include)
include_directories(third_party)

# Source files
set(CPU_SOURCES
    src/cpu/grayscale.c
    src/cpu/brightness.c
    src/cpu/invert.c
    src/cpu/gaussian_blur.c
    src/cpu/sobel.c
)

set(CUDA_SOURCES
    src/cuda/grayscale.cu
    src/cuda/brightness.cu
    src/cuda/invert.cu
    src/cuda/gaussian_blur.cu
    src/cuda/sobel.cu
)

set(UTILS_SOURCES
    src/utils/image_utils.c
    src/utils/benchmark.c
)

# Executable
add_executable(cuda_image_benchmark
    src/main.c
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(cuda_image_benchmark PRIVATE CUDA::cudart m)

# Compiler flags for optimization
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cuda_image_benchmark PRIVATE -O3 -Wall -Wextra)
endif()

if(CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    target_compile_options(cuda_image_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>)
endif()
