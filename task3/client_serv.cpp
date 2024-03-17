#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <future>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <functional>

template<typename T>
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

    size_t add_task(std::packaged_task<T()> task) {
        std::lock_guard<std::mutex> lock(tasks_mutex);
        std::future<T> result = task.get_future();  
        tasks.push(std::move(task));
        ind++;
        results[ind] = result.share();
        ids.push(ind);
        return ind;
    }

    double request_result(size_t id_res) {
        std::lock_guard<std::mutex> lock(results_mutex);
        return results[id_res].get();
    }

private:
    std::queue<std::packaged_task<T()>> tasks;
    std::queue<int> ids;
    std::unordered_map<size_t, std::shared_future<T>> results;
    std::mutex tasks_mutex;
    std::mutex results_mutex;
    std::thread server_thread;
    int ind;
    bool running;

    void run() {
        size_t id;
        std::packaged_task<T()> task;
        while (running) {
            if (!tasks.empty()) {
                
                std::lock_guard<std::mutex> lock(tasks_mutex);
                id = ids.front();
                task = std::move(tasks.front());
                task();
                tasks.pop();
                ids.pop();
            }
        }
    }
};


template<typename T>
class Client {
public:
    Client(Server<T>& server, size_t num_tasks, std::function<T(T, T)> task, std::string fp) 
    : server(server), num_tasks(num_tasks), task(task), fp(fp){}

    void run() {
        std::ofstream file(fp);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(1.0, 10.0);
        
        for (size_t i = 0; i < num_tasks; ++i) {
            T argument1 = dis(gen);
            T argument2 = dis(gen);
            std::packaged_task<T()> task_composed(std::bind(task, argument1, argument2));
            size_t id = server.add_task(std::move(task_composed));
            T ans = server.request_result(id);
            static std::mutex cout_mutex;
            std::lock_guard<std::mutex> lock(cout_mutex);
            file << "Arg1: " << argument1 << ", Arg2: " << argument2 << ": " << ans << std::endl;
        }

        file.close();
    }

private:
    Server<T>& server;
    size_t num_tasks;
    std::function<T(T, T)> task;
    std::string fp;
};

template<typename T>
T sin_task(T x, T) {
    return std::sin(x);
}

template<typename T>
T sqrt_task(T x, T) {
    return std::sqrt(x);
}

template<typename T>
T power_task(T base, T exponent) {
    return std::pow(base, exponent);
}

int main() {
    constexpr size_t num_tasks_per_client = 10;
    Server<double> server;
    server.start();

    Client<double> client1(server, num_tasks_per_client, sin_task<double>, "sin_results.txt");
    Client<double> client2(server, num_tasks_per_client, sqrt_task<double>, "sqrt_results.txt");
    Client<double> client3(server, num_tasks_per_client, power_task<double>, "power_results.txt");

    std::thread thread1(&Client<double>::run, &client1);
    std::thread thread2(&Client<double>::run, &client2);
    std::thread thread3(&Client<double>::run, &client3);

    thread1.join();
    thread2.join();
    thread3.join();

    server.stop();

    return 0;
}
