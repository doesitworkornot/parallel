#include <iostream>
#include <thread>

void helloFun()
{
    std::cout << "Hello from a function." << std::endl;
}

class HelloThread
{
public:
    void operator()() const
    {
        std::cout << "Hello from a functor." << std::endl;
    }
};

int main()
{
    std::thread t1(helloFun);

    HelloThread hello_class;
    std::thread t2(hello_class);

    std::thread t3([]{std::cout << "Hello from a lambda." << std::endl;});

    t1.join();
    t2.join();
    t3.join();

    std::cout << std::endl;
    return 0;
}