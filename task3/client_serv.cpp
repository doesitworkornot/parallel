#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <future>

// Контейнер для хранения результатов
#include <unordered_map>
#include <mutex>

// Подключаем пул потоков
#include "thread_pool.h"

class Server {
public:
    Server() {}

    void start() {
        running = true;
        server_thread = std::thread(&Server::run, this);
    }

    void stop() {
        running = false;
        server_thread.join();
    }

    size_t add_task(std::function<double()> task) {
        std::lock_guard<std::mutex> lock(tasks_mutex);
        tasks.emplace_back(std::move(task));
        ids.emplace_back(tasks.size()-1);
        return tasks.size()-1;
    }

    double request_result(size_t id_res) {
        std::lock_guard<std::mutex> lock(results_mutex);
        return results[id_res];
    }

private:
    std::vector<std::function<double()>> tasks;
    std::vector<double> ids;
    std::unordered_map<size_t, double> results;
    std::mutex tasks_mutex;
    std::mutex results_mutex;
    std::thread server_thread;
    bool running;

    void run() {
        ThreadPool pool(std::thread::hardware_concurrency());
        while (running) {
            while (!tasks.empty()) {
                size_t id;
                {
                    std::lock_guard<std::mutex> lock(tasks_mutex);
                    id = ids.back();
                    ids.pop_back();
                    auto task = tasks.back();
                    tasks.pop_back();
                    pool.add_task([this, task, id] {
                        double result = task();
                        std::lock_guard<std::mutex> lock(results_mutex);
                        results[id] = result;
                    });
                }
            }
        }
    }
};

class Client {
public:
    Client(Server& server, size_t num_tasks, std::function<double(double, double)> task, std::string fp) 
    : server(server), num_tasks(num_tasks), task(task), fp(fp){}

    void run() {
        std::ofstream file(fp);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(1.0, 10.0);

        for (size_t i = 0; i < num_tasks; ++i) {
            double argument1 = dis(gen);
            double argument2 = dis(gen);
            size_t id = server.add_task([this, argument1, argument2]() { return task(argument1, argument2); });
            double ans = server.request_result(id);
            static std::mutex cout_mutex;
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Got id: " << id << ", and ans is:" << ans << std::endl;
            file << "Result " << i << ": " << ans << std::endl;
        }

        file.close();
    }

private:
    Server& server;
    size_t num_tasks;
    std::function<double(double, double)> task;
    std::string fp;
};

// Задачи клиентов
double sin_task(double x, double) {
    return std::sin(x);
}

double sqrt_task(double x, double) {
    return std::sqrt(x);
}

double power_task(double base, double exponent) {
    return std::pow(base, exponent);
}

int main() {
    constexpr size_t num_tasks_per_client = 10;
    Server server;
    server.start();

    Client client1(server, num_tasks_per_client, sin_task, "sin_results.txt");
    Client client2(server, num_tasks_per_client, sqrt_task, "sqrt_results.txt");
    Client client3(server, num_tasks_per_client, power_task, "power_results.txt");

    std::thread thread1(&Client::run, &client1);
    std::thread thread2(&Client::run, &client2);
    std::thread thread3(&Client::run, &client3);

    thread1.join();
    thread2.join();
    thread3.join();

    server.stop();

    return 0;
}
