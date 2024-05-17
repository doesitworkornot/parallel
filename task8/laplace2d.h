class Laplace {
private:
    double* A, * Anew;
    double* hA;
    int m, n;

public:
    Laplace(int m, int n);
    ~Laplace();
    void initialize();
    void calcNext();
    double calcError();
    void swap();
};

__global__ void initializeKernel(double *A, double *Anew, int n, int m);
__global__ void calcNextKernel(double *A, double *Anew, int m, int n);
__global__ void calcErrorKernel(double *A, double *Anew, double &error, int n, int m, double *shared_error);