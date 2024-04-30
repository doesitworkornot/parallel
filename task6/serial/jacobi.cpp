#include <iostream>
#include <omp.h>
#include <new>
#include "laplace2d.h"
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char **argv) {
    int m = 4096;
    int iter_max = 1000;
    double tol = 1.0e-6;

    double error = 1.0;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("precision,p", po::value<double>(&tol)->default_value(1.0e-6), "precision")
        ("grid_size,n", po::value<int>(&m)->default_value(4096), "grid size")
        ("iterations,i", po::value<int>(&iter_max)->default_value(1000), "number of iterations");

    // Парсинг аргументов командной строки
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    double *restrict A = new double[m * m];
    double *restrict Anew = new double[m * m];

    nvtxRangePushA("init");
    initialize(A, Anew, m, m);
    nvtxRangePop();
    std::cout << "Jacobi relaxation Calculation: " <<  m  << " x "  << m << " mesh\n";

    std::cout << "Parameters: precision - " << tol << ", max iterations - " << iter_max << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;

    nvtxRangePushA("while");
    while (error > tol && iter < iter_max) {
        nvtxRangePushA("calc");
        error = calcNext(A, Anew, m, m);
        nvtxRangePop();

        nvtxRangePushA("swap");
        swap(A, Anew, m, m);
        nvtxRangePop();
        if (iter % 100 == 0)
            printf("%5d, %0.6f\n", iter, error);

        iter++;
    }
    nvtxRangePop();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_sec = std::chrono::duration_cast<std::chrono::seconds>(end - start);



    std::cout << " total: " << duration_sec.count() << "s\n";

    deallocate(A, Anew);

    return 0;
}