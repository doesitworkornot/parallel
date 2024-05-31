#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define OFFSET(x, y, m) (((x) * (m)) + (y))

namespace po = boost::program_options;

// Custom deleter for CUDA pointers
struct CudaDeleter {
    void operator()(double* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

struct CudaVoidDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

// Function to initialize arrays A and Anew
void initFunc(double* A, double* Anew, int n, int m);

// Kernel for matrix calculation
__global__ void Calculate_matrix(double* A, double* Anew, size_t size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size * size) return;

    int i = idx / size;
    int j = idx % size;

    if (!(j == 0 || i == 0 || j >= size - 1 || i >= size - 1)) {
        Anew[idx] = 0.25 * (A[idx - 1] + A[idx - size] + A[idx + size] + A[idx + 1]);
    }
}

// Kernel for error calculation
__global__ void Error_matrix(double* Anew, double* A, double* error, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size * size) return;

    error[idx] = fabs(A[idx] - Anew[idx]);
}

int main(int argc, char const* argv[]) {
    int m = 4096;
    int n;
    int iter_max = 1000;
    double tol = 1.0e-6;
    double error = 1.0;

    // Parsing program options
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
    n = m;

    // Allocating host memory
    auto A = std::make_unique<double[]>(n * m);
    auto Anew = std::make_unique<double[]>(n * m);
    memset(A.get(), 0, n * m * sizeof(double));
    memset(Anew.get(), 0, n * m * sizeof(double));
    initFunc(A.get(), Anew.get(), n, m);

    // CUDA stream and graph
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    // Allocating device memory using unique_ptr with custom deleters
    std::unique_ptr<double, CudaDeleter> error_GPU, error_device, Anew_device, A_device;
    double* raw_error_GPU;
    double* raw_error_device;
    double* raw_Anew_device;
    double* raw_A_device;

    cudaMalloc(&raw_error_GPU, sizeof(double) * n * n);
    cudaMalloc(&raw_error_device, sizeof(double) * n * n);
    cudaMalloc(&raw_Anew_device, sizeof(double) * n * n);
    cudaMalloc(&raw_A_device, sizeof(double) * n * n);

    error_GPU.reset(raw_error_GPU);
    error_device.reset(raw_error_device);
    Anew_device.reset(raw_Anew_device);
    A_device.reset(raw_A_device);

    // Copying data to device
    cudaMemcpy(A_device.get(), A.get(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Anew_device.get(), Anew.get(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Calculating size of array for device reduction and allocating it
    double* raw_tmp = nullptr;
    size_t tmp_size = 0;
    cub::DeviceReduce::Max(raw_tmp, tmp_size, Anew_device.get(), error_GPU.get(), n * n);
    cudaMalloc(&raw_tmp, tmp_size);
    std::unique_ptr<void, CudaVoidDeleter> tmp;
    tmp.reset(raw_tmp);

    int threads_per_block = 512;
    int num_blocks = (n * n + threads_per_block - 1) / threads_per_block;

    int iter = 0;
    auto start = std::chrono::high_resolution_clock::now();


    int iters_one_cycle = 1000;


    // Starting recording
    cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);

    for (int i = 0; i < iters_one_cycle; i++){
        Calculate_matrix<<<num_blocks, threads_per_block, 0, stream>>>(Anew_device.get(), A_device.get(), n);
        if (i==iters_one_cycle-1){
            Error_matrix<<<num_blocks, threads_per_block, 0, stream>>>(Anew_device.get(), A_device.get(), error_device.get(), n);
            cub::DeviceReduce::Max(tmp.get(),tmp_size,error_device.get(),error_GPU.get(),n*n,stream);
        }
        std::swap(A_device, Anew_device);
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    // End of recording

    // Iterating on the graph
    while (error > tol && iter < iter_max) {
        cudaGraphLaunch(graph_exec, stream);
        cudaMemcpy(&error, error_GPU.get(), sizeof(double), cudaMemcpyDeviceToHost);
        iter += iters_one_cycle;
        printf("%5d, %0.6f\n", iter, error);
    }

	auto end = std::chrono::high_resolution_clock::now();
    auto duration_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << " total: " << duration_sec.count() << " ms\n";

    cudaMemcpy(A.get(),A_device.get(),sizeof(double)*n*n,cudaMemcpyDeviceToHost);

    std::ofstream out("out.txt");
    out << std::fixed << std::setprecision(5);
    for (int j = 0; j < n; j++){
        for (int i = 0; i < m; i++){
            out << std::left << std::setw(10) << A[OFFSET(j, i, m)] << " ";
        }
        out << std::endl;
    }

    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graph_exec);

    return 0;
}

void initFunc(double* A, double* Anew, int n, int m) {
    double corners[4] = {10, 20, 30, 20};
    double step = (corners[1] - corners[0]) / (n - 1);

    int lastIdx = n - 1;
    A[0] = Anew[0] = corners[0];
    A[lastIdx] = Anew[lastIdx] = corners[1];
    A[n * lastIdx] = Anew[n * lastIdx] = corners[3];
    A[n * n - 1] = Anew[n * n - 1] = corners[2];

    for (int i = 1; i < n - 1; i++) {
        double val = corners[0] + i * step;
        A[i] = Anew[i] = val;
        A[n * i] = Anew[n * i] = val;
        A[lastIdx + n * i] = Anew[lastIdx + n * i] = corners[1] + i * step;
        A[n * lastIdx + i] = Anew[n * lastIdx + i] = corners[3] + i * step;
    }
}
