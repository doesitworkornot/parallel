#include <iostream>
#include <future>
#include <thread>
#include <chrono>


void getAnswer(std::promise<int> intPromise)
{
    using namespace std::literals::chrono_literals;
    std::this_thread::sleep_for(3s);
    intPromise.set_value(42);
}
int main()
{
    std::promise<int> answerPromise;
    auto fut = answerPromise.get_future();

    std::thread prodThread(getAnswer, std::move(answerPromise));

    std::future_status status{};
    do
    {   
        status = fut.wait_for(std::chrono::milliseconds(200));
        std::cout << "... doing something else" << std::endl;
    } while (status != std::future_status::ready);
    
    std::cout << std::endl;
    std::cout << "The Answer : " << fut.get() << '\n';
    prodThread.join();
}