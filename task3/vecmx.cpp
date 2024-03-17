#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <future>


void initializeMatrix(std::vector<double>& matrix, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i*n+j] = dis(gen);
        }
    }
}


void initializeVector(std::vector<double>& vector) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    for (auto& element : vector) {
        element = dis(gen);
    }
}


// Функция для выполнения умножения матрицы на вектор в заданном диапазоне строк
void partialMatrixVectorMultiplication(const std::vector<double>& matrix, const std::vector<double>& vector,
                                       std::vector<int>& result, size_t start, size_t end, int n) {
    for (size_t i = start; i < end; i++) {
        for (size_t j = 0; j < n; j++) {
            result[i] += matrix[i*n+j] * vector[j];
        }
    }
}

// Многопоточное умножение матрицы на вектор
std::vector<int> parallelMatrixVectorMultiplication(const std::vector<double>& matrix, const std::vector<double>& vector, int n) {
    std::vector<int> result(n);
    std::vector<std::thread> threads;

    size_t numThreads = std::thread::hardware_concurrency();
    size_t blockSize = n / numThreads;
    auto tick = std::chrono::steady_clock::now();
    size_t start = 0;
    
    for (size_t i = 0; i < numThreads; i++) {
        size_t end = start + blockSize;
        threads.emplace_back(partialMatrixVectorMultiplication, std::ref(matrix), std::ref(vector), std::ref(result), start, end, n);
        start = end;
    }
  
    threads.emplace_back(partialMatrixVectorMultiplication, std::ref(matrix), std::ref(vector), std::ref(result), start, n, n);
   
    for (auto& thread : threads) {
        thread.join();
    }
    auto tock = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    return result;
}

int main() {
    const size_t n = 40000;

    std::vector<double> matrix(n*n);
    std::vector<double> vector(n);

    auto future = std::async(std::launch::async, initializeMatrix, std::ref(matrix), n);

    auto sec_fut = std::async(std::launch::async, initializeVector, std::ref(vector));

    future.wait();
    sec_fut.wait();

    std::vector<int> result = parallelMatrixVectorMultiplication(matrix, vector, n);

    std::cout << std::endl;

    return 0;
}
