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



# 10 queen
```
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

int solution_count = 0;

bool is_safe(vector<vector<int>>& board, int row, int col, int n) {
    // Check if there's a queen in the same column
    for (int i = 0; i < row; ++i) {
        if (board[i][col] == 1)
            return false;
    }

    // Check upper left diagonal
    for (int i = row, j = col; i >= 0 && j >= 0; --i, --j) {
        if (board[i][j] == 1)
            return false;
    }

    // Check upper right diagonal
    for (int i = row, j = col; i >= 0 && j < n; --i, ++j) {
        if (board[i][j] == 1)
            return false;
    }

    return true;
}

bool solve_n_queens_util(vector<vector<int>>& board, int row, int n) {
    if (row == n) {
        // Print the solution
        for (const auto& r : board) {
            for (int val : r) {
                cout << val << " ";
            }
            cout << endl;
        }
        cout << endl;
        solution_count++;
        return true;
    }

    bool res = false;

    for (int col = 0; col < n; ++col) {
        if (is_safe(board, row, col, n)) {
            board[row][col] = 1;
            res = solve_n_queens_util(board, row + 1, n) || res;
            board[row][col] = 0;
        }
    }

    return res;
}

void solve_n_queens(int n) {
    vector<vector<int>> board(n, vector<int>(n, 0));

    auto start = high_resolution_clock::now();
    if (!solve_n_queens_util(board, 0, n)) {
        cout << "Solution does not exist" << endl;
    }
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Number of solutions: " << solution_count << endl;
    cout << "Time taken: " << duration.count() << " milliseconds" << endl;
}

int main() {
    int n = 10;
    solve_n_queens(n);
    return 0;
}
```
```
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

int solution_count = 0;

bool is_safe(const vector<vector<int>>& board, int row, int col, int n) {
    // Check if there's a queen in the same column
    for (int i = 0; i < row; ++i) {
        if (board[i][col] == 1)
            return false;
    }

    // Check upper left diagonal
    for (int i = row, j = col; i >= 0 && j >= 0; --i, --j) {
        if (board[i][j] == 1)
            return false;
    }

    // Check upper right diagonal
    for (int i = row, j = col; i >= 0 && j < n; --i, ++j) {
        if (board[i][j] == 1)
            return false;
    }

    return true;
}

void solve_n_queens_util(vector<vector<int>>& board, int row, int n) {
    if (row == n) {
#pragma omp critical
        {
            // Print the solution
            for (const auto& r : board) {
                for (int val : r) {
                    cout << val << " ";
                }
                cout << endl;
            }
            cout << endl;
            solution_count++;
        }

        return;
    }

    for (int col = 0; col < n; ++col) {
        if (is_safe(board, row, col, n)) {
            board[row][col] = 1;
            solve_n_queens_util(board, row + 1, n);
            board[row][col] = 0;
        }
    }
}

void solve_n_queens(int n) {
    vector<vector<int>> board(n, vector<int>(n, 0));

    auto start = high_resolution_clock::now();

#pragma omp parallel for
    for (int col = 0; col < n; ++col) {
        vector<vector<int>> local_board(board);
        local_board[0][col] = 1;
        solve_n_queens_util(local_board, 1, n);
    }

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Number of solutions: " << solution_count << endl;
    cout << "Time taken: " << duration.count() << " milliseconds" << endl;
}

int main() {
    int n = 10;
    solve_n_queens(n);
    return 0;
}
```


# place_horses_sequential
```
#include <iostream>
#include <vector>
using namespace std;

const int BOARD_SIZE = 10;

bool is_safe(const vector<vector<bool>>& board, int row, int col) {
    // Check if the cell is empty
    if (!board[row][col])
        return false;

    // Check if there's already a horse in the same row
    for (int j = 0; j < BOARD_SIZE; ++j) {
        if (j != col && board[row][j])
            return false;
    }

    // Check if there's already a horse in the same column
    for (int i = 0; i < BOARD_SIZE; ++i) {
        if (i != row && board[i][col])
            return false;
    }

    // Check for diagonals
    for (int i = row - 2, j = col - 1; i >= 0 && j >= 0; --i, --j) {
        if (board[i][j])
            return false;
    }

    for (int i = row - 2, j = col + 1; i >= 0 && j < BOARD_SIZE; --i, ++j) {
        if (board[i][j])
            return false;
    }

    for (int i = row + 2, j = col - 1; i < BOARD_SIZE && j >= 0; ++i, --j) {
        if (board[i][j])
            return false;
    }

    for (int i = row + 2, j = col + 1; i < BOARD_SIZE && j < BOARD_SIZE; ++i, ++j) {
        if (board[i][j])
            return false;
    }

    return true;
}

void place_horses_sequential(vector<vector<bool>>& board, int row) {
    if (row == BOARD_SIZE) {
        // Print the solution
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                cout << (board[i][j] ? "H " : "_ ");
            }
            cout << endl;
        }
        cout << endl;
        return;
    }

    for (int col = 0; col < BOARD_SIZE; ++col) {
        if (is_safe(board, row, col)) {
            board[row][col] = true;
            place_horses_sequential(board, row + 1);
            board[row][col] = false;
        }
    }
}

int main() {
    vector<vector<bool>> board(BOARD_SIZE, vector<bool>(BOARD_SIZE, true));
    place_horses_sequential(board, 0);
    return 0;
}
```


```
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

const int BOARD_SIZE = 10;

bool is_safe(const vector<vector<bool>>& board, int row, int col) {
    // Check if the cell is empty
    if (!board[row][col])
        return false;

    // Check if there's already a horse in the same row
    for (int j = 0; j < BOARD_SIZE; ++j) {
        if (j != col && board[row][j])
            return false;
    }

    // Check if there's already a horse in the same column
    for (int i = 0; i < BOARD_SIZE; ++i) {
        if (i != row && board[i][col])
            return false;
    }

    // Check for diagonals
    for (int i = row - 2, j = col - 1; i >= 0 && j >= 0; --i, --j) {
        if (board[i][j])
            return false;
    }

    for (int i = row - 2, j = col + 1; i >= 0 && j < BOARD_SIZE; --i, ++j) {
        if (board[i][j])
            return false;
    }

    for (int i = row + 2, j = col - 1; i < BOARD_SIZE && j >= 0; ++i, --j) {
        if (board[i][j])
            return false;
    }

    for (int i = row + 2, j = col + 1; i < BOARD_SIZE && j < BOARD_SIZE; ++i, ++j) {
        if (board[i][j])
            return false;
    }

    return true;
}

void place_horses_parallel(vector<vector<bool>>& board, int row) {
    if (row == BOARD_SIZE) {
        // Print the solution
#pragma omp critical
        {
            for (int i = 0; i < BOARD_SIZE; ++i) {
                for (int j = 0; j < BOARD_SIZE; ++j) {
                    cout << (board[i][j] ? "H " : "_ ");
                }
                cout << endl;
            }
            cout << endl;
        }
        return;
    }

#pragma omp parallel for
    for (int col = 0; col < BOARD_SIZE; ++col) {
        if (is_safe(board, row, col)) {
            board[row][col] = true;
            place_horses_parallel(board, row + 1);
            board[row][col] = false;
        }
    }
}

int main() {
    vector<vector<bool>> board(BOARD_SIZE, vector<bool>(BOARD_SIZE, true));
#pragma omp parallel for
    for (int col = 0; col < BOARD_SIZE; ++col) {
        board[0][col] = true;
        place_horses_parallel(board, 1);
        board[0][col] = false;
    }
    return 0;
}

```

