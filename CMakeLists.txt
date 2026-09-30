cmake_minimum_required(VERSION 3.20)

project(openmp_matmul_benchmark VERSION 0.1.0 LANGUAGES C)

add_executable(matmul_benchmark src/main.c)

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(matmul_benchmark PRIVATE -O3 -march=native -fopenmp)
    target_link_options(matmul_benchmark PRIVATE -fopenmp)
else()
    message(FATAL_ERROR "This initial version currently supports GCC only.")
endif()

