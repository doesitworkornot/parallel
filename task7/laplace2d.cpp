#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "laplace2d.h"
#include <omp.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define OFFSET(x, y, m) (((x) * (m)) + (y))

Laplace::Laplace(int m, int n) : m(m), n(n){
    A = new double[n * m];
    Anew = new double[n * m];
}


Laplace::~Laplace(){
    std::ofstream out("out.txt");
    out << std::fixed << std::setprecision(5);
    #pragma acc exit data copyout(A[:m*n], Anew[:m*n])
    for (int j = 0; j < n; j++){
        for (int i = 0; i < m; i++){
            out << std::left << std::setw(10) << A[OFFSET(j, i, m)] << " ";
        }
        out << std::endl;
    } 
    #pragma acc exit data delete (this)
    delete (A);
    delete (Anew);
}

void Laplace::initialize(){
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));

    double corners[4] = {10, 20, 30, 20};
    A[0] = corners[0];
    A[n - 1] = corners[1];
    A[n * n - 1] = corners[2];
    A[n * (n - 1)] = corners[3];
    Anew[0] = corners[0];
    Anew[n - 1] = corners[1];
    Anew[n * n - 1] = corners[2];
    Anew[n * (n - 1)] = corners[3];
    double step = (corners[1] - corners[0]) / (n - 1);


    for (int i = 1; i < n - 1; i ++) {
        A[i] = corners[0] + i * step;
        A[n * i] = corners[0] + i * step;
        A[(n-1) + n * i] = corners[1] + i * step;
        A[n * (n-1) + i] = corners[3] + i * step;
        Anew[i] = corners[0] + i * step;
        Anew[n * i] = corners[0] + i * step;
        Anew[(n-1) + n * i] = corners[1] + i * step;
        Anew[n * (n-1) + i] = corners[3] + i * step;
    }

    #pragma acc enter data copyin(this)
    #pragma acc enter data copyin(A[ : m * n], Anew[ : m * n])
}


void Laplace::calcNext(){
    #pragma acc parallel loop present(A, Anew)
    for (int j = 1; j < n - 1; j++){
        #pragma acc loop
        for (int i = 1; i < m - 1; i++){
            Anew[OFFSET(j, i, m)] = 0.25 * (A[OFFSET(j, i + 1, m)] + A[OFFSET(j, i - 1, m)] + A[OFFSET(j - 1, i, m)] + A[OFFSET(j + 1, i, m)]);
        }
    }
}

double Laplace::calcError() {
    double error = 0.0;

    double *d_A, *d_Anew, *d_error;
    cudaMalloc(&d_A, m * n * sizeof(double));
    cudaMalloc(&d_Anew, m * n * sizeof(double));
    cudaMalloc(&d_error, m * n * sizeof(double));

    cudaMemcpy(d_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, Anew, m * n * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = -1.0;
    cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, d_Anew, m, &alpha, d_A, m, d_error, m);

    int* max_ind;
    cublasIdamax(handle, m * n, d_error, 1, max_ind);

    int* min_ind;
    cublasIdamin(handle, m * n, d_error, 1, min_ind);
    error = fmax(fabs(*min_ind), *max_ind);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_Anew);
    cudaFree(d_error);
    return error;
}
void Laplace::swap(){
    double* temp = A;
    A = Anew;
    Anew = temp;
    #pragma acc data present(A, Anew)
    return;
}