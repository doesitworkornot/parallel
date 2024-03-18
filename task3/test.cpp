#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <iomanip>
#include <thread>


template<typename T>
T power_task(T base, T exponent) {
    return std::pow(base, exponent);
}

template<typename T>
T sqrt_task(T x, T) {
    return std::sqrt(x);
}

template<typename T>
T sin_task(T x, T) {
    return std::sin(x);
}

void sin_test(std::string fp){
    std::ifstream file(fp);
    int flag = 1;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            double arg1, arg2, ans, result;
            std::istringstream iss(line);
            iss >> arg1 >> arg2 >> ans;
            result = sin_task(arg1, arg2);
            if (std::fabs(result - ans) > 1e-5) {
                std::cout << fp << " - Failed" << std::endl;
                std::cout << arg1 << ' ' << arg2<< ' '  << ans<< ' '  << result << ' ' << std::endl;
                flag = 0;
                break;
            }   
        }
        file.close();
        if (flag)
            std::cout << fp << " - Passed" << std::endl;
    }  
}

void sqrt_test(std::string fp){
    std::ifstream file(fp);
    int flag = 1;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            double arg1, arg2, ans, result;
            std::istringstream iss(line);
            iss >> arg1 >> arg2 >> ans;
            result = sqrt_task(arg1, arg2);
            if (std::fabs(result - ans) > 1e-5) {
                std::cout << fp << " - Failed" << std::endl;
                std::cout << arg1 << ' ' << arg2<< ' '  << ans<< ' '  << result << ' ' << std::endl;
                flag = 0;
                break;
            }   
        }
        file.close();
        if (flag)
            std::cout << fp << " - Passed" << std::endl;
    } 
}

void pow_test(std::string fp){
    std::ifstream file(fp);
    int flag = 1;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            double arg1, arg2, ans, result;
            std::istringstream iss(line);
            iss >> arg1 >> arg2 >> ans;
            result = power_task(arg1, arg2);
            if (std::fabs(result - ans) > 1e-5) {
                std::cout << fp << " - Failed" << std::endl;
                std::cout << arg1 << ' ' << arg2<< ' '  << ans<< ' '  << result << ' ' << std::endl;
                flag = 0;
                break;
            }   
        }
        file.close();
        if (flag)
            std::cout << fp << " - Passed" << std::endl;
    } 
}

int main() {

    std::thread thread1(sin_test, "sin_results.txt");
    std::thread thread2(sqrt_test, "sqrt_results.txt");
    std::thread thread3(pow_test, "power_results.txt");


    thread1.join();
    thread2.join();
    thread3.join();

    return 0;
}
