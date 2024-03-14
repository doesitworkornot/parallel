#include <chrono>
#include <mutex>
#include <thread>
#include <iostream>
#include <vector>

std::mutex mut;

void workOnResource()
{
    std::lock_guard<std::mutex> lock{mut};
    std::cout << "lock \n";
    std::cout << "id thread " << std::this_thread::get_id() << '\n';
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

void workOnResource_unique()
{
    std::unique_lock<std::mutex> u_lock{mut, std::defer_lock};
    u_lock.lock();
    std::cout << "lock \n";
    std::cout << "id thread " << std::this_thread::get_id() << '\n';
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    u_lock.unlock();
}

int main()
{
    std::vector<std::thread> threads;
    size_t num_threads = 10;

    for (size_t i = 0; i < num_threads; ++i)
    {
        threads.push_back(std::thread(workOnResource));
    }

    for (auto& thr: threads)
    {
        thr.join();
    }
    
    return 0;
}

