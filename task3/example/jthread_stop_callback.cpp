#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>



auto func = [](std::stop_token stoken)
{
    using namespace std::literals;
    std::atomic<int> counter{0};
    auto thread_id = std::this_thread::get_id();
    std::stop_callback callBack(stoken, [&counter, thread_id] {
                                std::cout << "Thread id : " << thread_id 
                                << "; counter: " << counter << '\n';
                                });
    while (counter < 10)
    {
        std::this_thread::sleep_for(0.2s);
        ++counter;
    }
};

int main()
{
    std::cout << '\n';
    std::vector<std::jthread> vecThreads(10);

    for (auto &thr : vecThreads)
        thr = std::jthread(func);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    for (auto &thr : vecThreads)
        thr.request_stop();

    std::cout << '\n';
}