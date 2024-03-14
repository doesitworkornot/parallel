#include <iostream>
#include <map>
#include <thread>
#include <shared_mutex>
#include <string>
#include <chrono>

std::map<std::string, int> telebook{
    {"Scott", 1}, {"Ritchie", 9}, {"Dijkstra", 22}};

std::shared_mutex telebookMutex;

void addTeleBook(const std::string &name, int number)
{
    std::lock_guard<std::shared_mutex> writeLock(telebookMutex);
    std::cout << "STARTING IPDATE" << name;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    telebook[name] = number;
    std::cout << "... ENDING UPDATE " << name << std::endl;
}

void printNumber(const std::string &name)
{
    std::shared_lock<std::shared_mutex> readerLock(telebookMutex);
    auto searchEntry = telebook.find(name);
    if (searchEntry != telebook.end())
    {
        std::cout << searchEntry->first << ": " << searchEntry->second << std::endl;
    }
    else
    {
        std::cout << name << " not found!" << std::endl;
    }
}

int main()
{
    std::thread reader1([]
                        { printNumber("Scott"); });
    std::thread reader2([]
                        { printNumber("Ritchie"); });
    std::thread w1([]
                   { addTeleBook("Scott", 1968); });
    std::thread reader3([]
                        { printNumber("Dijkstra"); });
    std::thread reader4([]
                        { printNumber("Scott"); });
    std::thread w2([]
                   { addTeleBook("Bjarne", 1965); });
    std::thread reader5([]
                        { printNumber("Scott"); });
    std::thread reader6([]
                        { printNumber("Ritchie"); });
    std::thread reader7([]
                        { printNumber("Scott"); });
    std::thread reader8([]
                        { printNumber("Bjarne"); });

    reader1.join();
    reader2.join();
    reader3.join();
    reader4.join();
    reader5.join();
    reader6.join();
    reader7.join();
    reader8.join();

    w1.join();
    w2.join();

    std::cout << std::endl;
    std::cout << "\nThe new telephone book" << std::endl;

    for (auto teleIt : telebook)
    {

        std::cout << teleIt.first << ": " << teleIt.second << std::endl;
    }
}