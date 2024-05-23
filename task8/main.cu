#include <iostream>
#include <omp.h>
#include <new>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <boost/program_options.hpp>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <cub/cub.cuh>
#include <algorithm> 

#define OFFSET(x, y, m) (((x) * (m)) + (y))

class Laplace {
private:
    double* A, * Anew;
    double* hA;
    int m, n;

public:
    Laplace(int m, int n);
    ~Laplace();
    void calcNext();
    double calcError();
    void swap();
};

Laplace::Laplace(int m, int n) : m(m), n(n){
    cudaMalloc(&A, m * n * sizeof(double));
    cudaMalloc(&Anew, m * n * sizeof(double));
    int blockSize = 256;
    int numBlocks = (n * m + blockSize - 1) / blockSize;
    initializeKernel<<<numBlocks, blockSize>>>(A, Anew, n, m);
    cudaDeviceSynchronize();
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

__global__ void calcErrorKernel(const double *A, const double *Anew, int n, int m, double *d_errors) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_elements = (n - 2) * (m - 2);

    if (idx < total_elements) {
        int j = 1 + idx / (m - 2);
        int i = 1 + idx % (m - 2);
        d_errors[idx] = fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i, m)]);
    }
}

double Laplace::calcError() {
    int total_elements = (n - 2) * (m - 2);
    double *d_errors = nullptr;
    cudaMalloc(&d_errors, total_elements * sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    calcErrorKernel<<<blocksPerGrid, threadsPerBlock>>>(A, Anew, n, m, d_errors);

    double *d_maxError = nullptr;
    cudaMalloc(&d_maxError, sizeof(double));

    // Allocate temporary storage
    void *d_temp_storage = nullptr;
    size_t d_temp_storage_bytes = 0;

    // Obtain the required temporary storage size
    cub::DeviceReduce::Max(d_temp_storage, d_temp_storage_bytes, d_errors, d_maxError, total_elements);
    cudaMalloc(&d_temp_storage, d_temp_storage_bytes);

    // Run the reduction to find the maximum error
    cub::DeviceReduce::Max(d_temp_storage, d_temp_storage_bytes, d_errors, d_maxError, total_elements);

    double h_maxError;
    cudaMemcpy(&h_maxError, d_maxError, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_errors);
    cudaFree(d_maxError);
    cudaFree(d_temp_storage);

    return h_maxError;
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

namespace po = boost::program_options;

int main(int argc, char **argv) {
    int m = 4096;
    int iter_max = 1000;
    double tol = 1.0e-6;

    double error = 1.0;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Usage: set precision by running -p 123, set grid size by using -n 123 and set maximum number of iterations by -i 123")
        ("precision,p", po::value<double>(&tol)->default_value(1.0e-6), "precision")
        ("grid_size,n", po::value<int>(&m)->default_value(4096), "grid size")
        ("iterations,i", po::value<int>(&iter_max)->default_value(1000), "number of iterations");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    
    nvtxRangePushA("init");
    Laplace a(m, m);
    nvtxRangePop();
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", m, m);

    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;

    nvtxRangePushA("while");
    while (error > tol && iter < iter_max)
    {
        nvtxRangePushA("calc");
        a.calcNext();
        nvtxRangePop();
        
        if (iter % 1000 == 0){
            nvtxRangePushA("error");
            error = a.calcError();
            nvtxRangePop();
            printf("%5d, %0.6f\n", iter, error);
        }
        
        nvtxRangePushA("swap");
        a.swap();
        nvtxRangePop();

        iter++;
    }
    nvtxRangePop();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);



    std::cout << " total: " << duration_sec.count() << "s\n";

    return 0;
}