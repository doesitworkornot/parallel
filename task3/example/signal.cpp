#include <iostream> 
#include <csignal>
 
void signal_handler(int signal)
{
    std::cout << "Div zero\n";
    std::exit(1);
}

int main() 
{ 
    std::signal(SIGFPE, signal_handler);

    try { 
        int result = 1 / 0; // Attempting division by zero 
        std::cout << "Result: " << result << std::endl; 
    } 
    catch (const std::exception& e) { 
        std::cout << "Exception caught: " << e.what()  << std::endl; 
    } 
  
    return 0; 
}