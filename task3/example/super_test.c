#include <stdio.h>
#include <unistd.h>

int main(){
    int a = 0;
    printf("%p", &a);
    usleep(10000000);
    return 0;
}