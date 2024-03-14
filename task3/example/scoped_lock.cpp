#include <chrono>
#include <thread>
#include <iostream>
#include <mutex>

struct CriticalData
{
    std::mutex mut;
};

void deadlock(CriticalData &a, CriticalData &b)
{
    std::scoped_lock scop_lock(a.mut, b.mut);
    std::cout << "One lock\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::cout << "Two lock\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

int main()
{
    CriticalData a;
    CriticalData b;

    std::thread t1([&]{deadlock(a,b);});
    std::thread t2([&]{deadlock(b,a);});

    t1.join();
    t2.join();

    return 0;
}