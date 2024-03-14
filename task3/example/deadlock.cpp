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
    a.mut.lock();
    std::cout << "One lock\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    b.mut.lock();
    std::cout << "Two lock\n";
    a.mut.unlock();
    b.mut.unlock();
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