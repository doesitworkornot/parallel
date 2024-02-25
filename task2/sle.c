#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>


double cpuSecond(){
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void initialize_matrix(double** matrix, int size){
    #pragma omp for
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            if (i!=j)
                matrix[i][j] = 1.0;
            else
                matrix[i][j] = 2.0;
        }
    }
}

void mx_vec_mult(double** mx, double* vec, double* res, int size){
    #pragma omp for
    for (int i = 0; i < size; i++){
        int sum = 0;
        for(int j = 0; j<size; j++){
            sum += mx[i][j] * vec[j];
        }
        res[i] = sum;
    }
}

void vec_subtraction(double* vec_1, double* vec_2, double* res, int size){
    #pragma omp for
    for (int i = 0; i < size; i++){
        res[i] = vec_1[i] - vec_2[i];
    }
}

void inplace_vec_subtraction(double* vec_1, double* vec_2, int size){
#pragma omp for
  for (int i = 0; i < size; i++){
        vec_1[i] -= vec_2[i];
    }
}

void scale_by(double* vec, double scale, int size){
    #pragma omp for
    for (int i = 0; i < size; i++){
        vec[i] *= scale;
    }
}

double find_norm(double* vec, int size){
    double norm = 0;
        double sumloc = 0.0;
        #pragma omp for
        for (int i = 0; i < size; i++){
            norm += pow(vec[i], 2);
        }
        #pragma omp atomic
        norm += sumloc;
    return sqrt(norm);
}

void simple_iteration(double** matrix, double* vec_x, double* vec_b, int size, double tau, double eps){
    double* ax = (double*)malloc(size*sizeof(double));
    double* subtracted = (double*)malloc(size*sizeof(double));
    double norm_b = find_norm(vec_b, size);
    double norm_sub;

    double t = cpuSecond();
        #pragma omp parallel
    {
    while(1){
    
        mx_vec_mult(matrix, vec_x, ax, size);
        vec_subtraction(ax, vec_b, subtracted, size);
        norm_sub = find_norm(subtracted, size);
        if (norm_sub / norm_b < eps)
            break;
        scale_by(subtracted, tau, size);
        inplace_vec_subtraction(vec_x, subtracted, size);
    }
    }
    t = cpuSecond() - t;

    printf("Elapsed time (serial): %.6f sec.\n", t);
}

int main(int argc, char** argv){
    int size = 1000;
    if (argc > 1)
        size = atoi(argv[1]);

    double tau = 0.001;
    if (argc > 2)
        tau = atof(argv[2]);
    
    double eps = 0.00001;
    if (argc > 3)
        eps = atof(argv[3]);
    
    double** matrix = (double**)calloc(size, sizeof(double*));
    for (int i = 0; i < size; i++)
        matrix[i] = (double*)calloc(size, sizeof(double));

    double* vec_x = (double*)calloc(size, sizeof(double));

    double* vec_b = (double*)malloc(size*sizeof(double));
    for (int i = 0; i < size; i++){
        vec_b[i] = size+1;
    }
    initialize_matrix(matrix, size);

    simple_iteration(matrix, vec_x, vec_b, size ,tau, eps);

    double sum = 0;

    for(int i = 0; i<size; i++){
        sum+=vec_x[i];
    }

    printf("Control avg value: %lf\n", sum/size);

    return 0;
}