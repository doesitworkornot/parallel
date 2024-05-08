cmake_minimum_required(VERSION 3.22)

project(Example_class VERSION 1.0 LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_COMPILER "pgc++")
set(NAME "super.cpp")

list(APPEND option_compile -acc=gpu -Minfo=all)
list(APPEND option_link -acc=gpu)

message("Compile C++: " ${CMAKE_CXX_COMPILER})

find_package(CUDA REQUIRED)
message("Lib" ${CUDA_INCLUDE_DIRS})

add_executable(${NAME} "example_cublas.c") 
target_compile_features(${NAME} PRIVATE cxx_std_20)
# openACC
target_compile_options(${NAME} PRIVATE ${option_compile}) 
target_link_options(${NAME} PRIVATE ${option_link})
# CUDA and cuBLAS
target_include_directories(${NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${NAME} CUDA::cublas)
# Include Boost Program Options
target_link_libraries(${PROJECT_NAME} Boost::program_options)