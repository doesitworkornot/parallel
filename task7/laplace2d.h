class Laplace {
private:
    double* A, * Anew;
    int m, n;

public:
    Laplace(int m, int n);
    ~Laplace();
    void calcNext();
    double calcError();
    void swap();
};