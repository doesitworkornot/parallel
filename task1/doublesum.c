#include <stdio.h>
#include <math.h> 
#include <stdlib.h>
#ifndef  M_PI
#define  M_PI  3.1415926535897932384626433
#endif

int main(){
    int to = 10000000;
    double sum = 0;
    double frac = 1/(double)to;
    double* arr = (double*)malloc(to * sizeof(double));
    double val;
    double arg;

    for (int i = 0; i < to; i++){
        arg = i * frac * 2 * M_PI;
        val = sin(arg);
        sum += val;
        arr[i] = val;
        if (i%10000 == 1){
            printf("%lf  %lf\n", val, arg);
        }
    }
    printf("Result is %lf using double\n", sum);

    return 0;
}