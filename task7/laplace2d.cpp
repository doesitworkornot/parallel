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


void Laplace::calcNext(){
    #pragma acc parallel loop present(A, Anew)
    for (int j = 1; j < n - 1; j++){
        #pragma acc loop
        for (int i = 1; i < m - 1; i++){
            Anew[OFFSET(j, i, m)] = 0.25 * (A[OFFSET(j, i + 1, m)] + A[OFFSET(j, i - 1, m)] + A[OFFSET(j - 1, i, m)] + A[OFFSET(j + 1, i, m)]);
        }
    }
}

double Laplace::calcError(){
    cublasHandle_t handler;
	cublasStatus_t status;

	status = cublasCreate(&handler);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasCreate failed with error code: " << status << std::endl;
        return 13;
    }	
	double error = 1.0;
	int idx = 0;
	double alpha = -1.0;
    #pragma acc data present (A, Anew) wait
    #pragma acc host_data use_device(A, Anew)
    {
        status = cublasDaxpy(handler, n * n, &alpha, Anew, 1, A, 1);

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasDaxpy failed with error code: " << status << std::endl;
            exit (13);
        }

                
        status = cublasIdamax(handler, n * n, A, 1, &idx);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasIdamax failed with error code: " << status << std::endl;
            exit (13);
        }
    }

    #pragma acc update host(A[idx - 1]) 
    error = std::fabs(A[idx - 1]);

    #pragma acc host_data use_device(A, Anew)
    status = cublasDcopy(handler, n * n, Anew, 1, A, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasDcopy failed with error code: " << status << std::endl;
        exit (13);
    }
    return error;
}

void Laplace::swap(){
    double* temp = A;
    A = Anew;
    Anew = temp;
    #pragma acc data present(A, Anew)
    return;
}


void printMatrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}