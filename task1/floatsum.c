#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#ifndef  M_PI
#define  M_PI  3.1415926535897932384626433
#endif 

int main(){
    int to = 10000000;
    float sum = 0;
    float frac = 1/(float)to;
    float* arr = (float*)malloc(to * sizeof(float));
    float val;
    float arg;

    for (int i = 0; i < to; i++){
        arg = i * frac * 2 * M_PI;
        val = (float)sin(arg);
        sum += val;
        arr[i] = val;
        if (i%10000 == 1){
            printf("%f  %f\n", val, arg);
        }
    }
    printf("Result is %f using float\n", sum);
    return 0;
}
