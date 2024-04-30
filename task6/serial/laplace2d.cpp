#include <cmath>
#include <cstring>

#define OFFSET(x, y, m) (((x)*(m)) + (y))

void initialize(double *restrict A, double *restrict Anew, int m, int n) {
    std::memset(A, 0, n * m * sizeof(double));
    std::memset(Anew, 0, n * m * sizeof(double));

    for (int i = 0; i < m; i++) {
        A[i] = 1.0;
        Anew[i] = 1.0;
    }
}

double calcNext(double *restrict A, double *restrict Anew, int m, int n) {
    double error = 0.0;
    for (int j = 1; j < n - 1; j++) {
        for (int i = 1; i < m - 1; i++) {
            Anew[OFFSET(j, i, m)] = 0.25 * (A[OFFSET(j, i + 1, m)] + A[OFFSET(j, i - 1, m)] +
                                            A[OFFSET(j - 1, i, m)] + A[OFFSET(j + 1, i, m)]);
            error = std::fmax(error, std::fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i, m)]));
        }
    }
    return error;
}

void swap(double *restrict A, double *restrict Anew, int m, int n) {
    for (int j = 1; j < n - 1; j++) {
        for (int i = 1; i < m - 1; i++) {
            A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];
        }
    }
}

void deallocate(double *restrict A, double *restrict Anew) {
    delete[] A; 
    delete[] Anew;
}