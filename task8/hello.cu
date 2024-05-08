#include "cuda_runtime.h"
#include <stdio.h>

__global__ void HelloWorld(){
    printf("hello world, %d, %d\n", blockIdx.x, threadIdx.x);
}

int main(){
    HelloWorld<<<100, 100>>>();
    cudaDeviceSynchronize();
    return 0;
}