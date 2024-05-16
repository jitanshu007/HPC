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


