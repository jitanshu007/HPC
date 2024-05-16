# HPC

# matrix_vector_multiply
```
 #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
```

# matrix_matrix_multiply
```
 #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
```

# vector_add
```
  #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
```

# vector_cross_product
```
  #pragma omp parallel sections
    {
        #pragma omp section
        C[0] = A[1] * B[2] - A[2] * B[1];

        #pragma omp section
        C[1] = A[2] * B[0] - A[0] * B[2];

        #pragma omp section
        C[2] = A[0] * B[1] - A[1] * B[0];
    }
```

# Jacobi Method:
```
#include <stdio.h>
#include <omp.h>

#define N 100
#define MAX_ITER 1000
#define TOLERANCE 1e-6

void jacobi(double A[N][N], double b[N], double x[N]) {
    int i, j, k;
    double sum;
    double x_new[N];

    for (k = 0; k < MAX_ITER; k++) {
        #pragma omp parallel for private(i, j, sum)
        for (i = 0; i < N; i++) {
            sum = 0.0;
            for (j = 0; j < N; j++) {
                if (j != i) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }

        double max_diff = 0.0;
        #pragma omp parallel for private(i)
        for (i = 0; i < N; i++) {
            double diff = fabs(x[i] - x_new[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            x[i] = x_new[i];
        }

        if (max_diff < TOLERANCE) {
            break;
        }
    }
}

int main() {
    double A[N][N];
    double b[N];
    double x[N];

    // Initialize A, b, x

    jacobi(A, b, x);

    // Print or use x

    return 0;
}
```


# Lu
```
#include <stdio.h>
#include <omp.h>

#define N 100

void luFactorization(double A[N][N], double L[N][N], double U[N][N]) {
    int i, j, k;
    
    // Initialize L and U to zero matrices
    #pragma omp parallel for private(i, j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }

    // LU decomposition
    for (i = 0; i < N; i++) {
        // Upper triangular matrix
        #pragma omp parallel for private(j)
        for (j = i; j < N; j++) {
            U[i][j] = A[i][j];
            for (k = 0; k < i; k++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
        
        // Lower triangular matrix
        #pragma omp parallel for private(j)
        for (j = i; j < N; j++) {
            if (i == j) {
                L[i][i] = 1.0;
            } else {
                L[j][i] = A[j][i];
                for (k = 0; k < i; k++) {
                    L[j][i] -= L[j][k] * U[k][i];
                }
                L[j][i] /= U[i][i];
            }
        }
    }
}

int main() {
    double A[N][N];
    double L[N][N];
    double U[N][N];

    // Initialize A

    luFactorization(A, L, U);

    // Print or use L and U

    return 0;
}
```

# Odd-even transposition sort
```
void odd_even_sort(vector<int>& arr) {
    int n = arr.size();
    #pragma omp parallel for default(none) shared(arr, n)
    for (int i = 0; i < n; ++i) {
        for (int j = i % 2; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}
```


# Quicksort
```
void quick_sort(vector<int>& arr) {
    if (arr.size() <= 1)
        return;
    int pivot = arr[arr.size() / 2];
    vector<int> left, middle, right;
    for (int num : arr) {
        if (num < pivot)
            left.push_back(num);
        else if (num > pivot)
            right.push_back(num);
        else
            middle.push_back(num);
    }
    #pragma omp task default(none) shared(left)
    quick_sort(left);
    #pragma omp task default(none) shared(right)
    quick_sort(right);
    #pragma omp taskwait
    copy(left.begin(), left.end(), arr.begin());
    copy(middle.begin(), middle.end(), arr.begin() + left.size());
    copy(right.begin(), right.end(), arr.begin() + left.size() + middle.size());
}
```


# Bitonic sort
```
void bitonic_merge(vector<int>& arr, int low, int cnt, bool direction) {
    if (cnt > 1) {
        int k = cnt / 2;
        #pragma omp parallel for default(none) shared(arr, low, k, cnt, direction)
        for (int i = low; i < low + k; ++i) {
            if ((arr[i] > arr[i + k]) == direction) {
                swap(arr[i], arr[i + k]);
            }
        }
        bitonic_merge(arr, low, k, direction);
        bitonic_merge(arr, low + k, k, direction);
    }
}

void bitonic_sort_recursive(vector<int>& arr, int low, int cnt, bool direction) {
    if (cnt > 1) {
        int k = cnt / 2;
        #pragma omp task default(none) shared(arr, low, k, direction)
        bitonic_sort_recursive(arr, low, k, true);
        #pragma omp task default(none) shared(arr, low, k, direction)
        bitonic_sort_recursive(arr, low + k, k, false);
        #pragma omp taskwait
        bitonic_merge(arr, low, cnt, direction);
    }
}

void bitonic_sort(vector<int>& arr) {
    bitonic_sort_recursive(arr, 0, arr.size(), true);
}
```



