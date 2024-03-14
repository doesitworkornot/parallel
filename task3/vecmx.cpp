#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <future>


void initializeMatrix(std::vector<std::vector<double>>& matrix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    for (auto& row : matrix) {
        for (auto& element : row) {
            element = dis(gen);
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
void partialMatrixVectorMultiplication(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector,
                                       std::vector<int>& result, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

// Многопоточное умножение матрицы на вектор
std::vector<int> parallelMatrixVectorMultiplication(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector) {
    std::vector<int> result(matrix.size(), 0);
    std::vector<std::thread> threads;

    size_t numThreads = std::thread::hardware_concurrency(); //
    size_t blockSize = matrix.size() / numThreads;
    auto tick = std::chrono::steady_clock::now();
    size_t start = 0;
    for (size_t i = 0; i < numThreads - 1; ++i) {
        size_t end = start + blockSize;
        threads.emplace_back(partialMatrixVectorMultiplication, std::ref(matrix), std::ref(vector), std::ref(result), start, end);
        start = end;
    }

    threads.emplace_back(partialMatrixVectorMultiplication, std::ref(matrix), std::ref(vector), std::ref(result), start, matrix.size());

    for (auto& thread : threads) {
        thread.join();
    }
    auto tock = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick);
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    return result;
}

int main() {
    const size_t rows = 40000;
    const size_t cols = 40000;

    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    std::vector<double> vector(cols);

    auto future = std::async(std::launch::async, initializeMatrix, std::ref(matrix));

    auto sec_fut = std::async(std::launch::async, initializeVector, std::ref(vector));

    future.wait();
    sec_fut.wait();

    std::cout << std::endl;

    std::vector<int> result = parallelMatrixVectorMultiplication(matrix, vector);

    std::cout << std::endl;

    return 0;
}
