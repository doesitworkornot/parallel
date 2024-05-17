#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "laplace2d.h"
#include <omp.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define OFFSET(x, y, m) (((x) * (m)) + (y))

Laplace::Laplace(int m, int n) : m(m), n(n){
    cudaMalloc(&A, m * n * sizeof(double));
    cudaMalloc(&Anew, m * n * sizeof(double));
}


Laplace::~Laplace() {
    std::ofstream out("out.txt");
    out << std::fixed << std::setprecision(5);
    hA = new double[n * m];
    cudaMemcpy(hA, A, n * m * sizeof(double), cudaMemcpyDeviceToHost);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            out << std::left << std::setw(10) << hA[OFFSET(j, i, m)] << " ";
        }
        out << std::endl;
    }
    out.close();
    cudaFree(A);
    cudaFree(Anew);
    delete (hA);
}

__global__ void initializeKernel(double *A, double *Anew, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double corners[4] = {10, 20, 30, 20};
    if (i < n * m) {
        int row = i / n;
        int col = i % n;
        double step = (n == m) ? (m - 1) / (corners[1] - corners[0]) : 1;
        double corners[4] = {10, 20, 30, 20};

        if (row == 0) {
            A[i] = corners[0];
            Anew[i] = corners[0];
        } else if (row == n - 1) {
            A[i] = corners[1];
            Anew[i] = corners[1];
        } else if (col == 0) {
            A[i] = corners[3] + row * step;
            Anew[i] = corners[3] + row * step;
        } else if (col == m - 1) {
            A[i] = corners[2] + row * step;
            Anew[i] = corners[2] + row * step;
        } else {
            A[i] = corners[0] + col * step;
            Anew[i] = corners[0] + col * step;
        }
    }
}

void Laplace::initialize() {
    int blockSize = 256;
    int numBlocks = (n * m + blockSize - 1) / blockSize;
    initializeKernel<<<numBlocks, blockSize>>>(A, Anew, n, m);
    cudaDeviceSynchronize();
}

__global__ void calcNextKernel(double *A, double *Anew, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < m - 1 && j < n - 1) {
        Anew[j * m + i] = 0.25 * (A[j * m + (i + 1)] + A[j * m + (i - 1)] + A[(j - 1) * m + i] + A[(j + 1) * m + i]);
    }
}

void Laplace::calcNext() {
    dim3 block_size(16, 16); 
    dim3 grid_size((m + block_size.x - 2) / block_size.x, (n + block_size.y - 2) / block_size.y);

    calcNextKernel<<<grid_size, block_size>>>(A, Anew, m, n);

    cudaDeviceSynchronize();
}

constexpr int THREADS_PER_BLOCK = 256; // Define the number of threads per block

double Laplace::calcError() {
    double error = 0.0;

    // Declare shared memory for block-level reduction
    __shared__ double shared_error[THREADS_PER_BLOCK];

    // Calculate grid and block dimensions
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((n - 2 + block.x - 1) / block.x);

    // Kernel for error calculation
    calcErrorKernel<<<grid, block>>>(A, Anew, error, n, m, shared_error);

    // Reduction across block-level errors
    double block_max_error;
    cub::BlockReduce<double>(shared_error).Reduce(block_max_error, error, cub::Max(), block.x);

    return block_max_error;
}

__global__ void calcErrorKernel(double *A, double *Anew, double &error, int n, int m, double *shared_error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (n - 2) * (m - 2)) {
        int j = i / (m - 2) + 1;
        int k = i % (m - 2) + 1;

        double diff = fabs(Anew[OFFSET(j, k, m)] - A[OFFSET(j, k, m)]);

        // Store thread-level errors to shared memory
        shared_error[threadIdx.x] = diff;

        // Synchronize threads to ensure all shared memory values are populated
        __syncthreads();

        // Perform block-level reduction
        cub::BlockReduce<double>(shared_error).Reduce(error, cub::Max(), block.x);

        // The first thread of each block updates the global error
        if (threadIdx.x == 0) {
            atomicMax(reinterpret_cast<unsigned long long*>(&error), __double_as_longlong(shared_error[0]));
        }
    }
}

void Laplace::swap() {
    double *d_temp;
    cudaMalloc(&d_temp, n * m * sizeof(double));
    cudaMemcpy(d_temp, A, n * m * sizeof(double), cudaMemcpyDeviceToDevice);
    double *temp = A;
    A = Anew;
    Anew = temp;
    cudaFree(d_temp);
}