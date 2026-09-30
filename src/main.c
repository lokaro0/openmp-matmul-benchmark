/*
実行結果 : OK (max relative error = 0.000e+00)
           n = 1000, 1.300e-01 [sec], 1.539e+01 [GFlops], speedup = 1.181e+01 [times]
使用したCPU : Intel 13th Core i7-1360P
gcc バージョン : 14.2.0
コンパイラオプション : -O3 -march=native -fopenmp -lpthread
使用した技法について : 
ループ順序の交換
ループの順番を変更し、行列Bと行列Cに対するメモリアクセスが行方向とした。
これにより、CPUのキャッシュメモリを効率的に利用できるようになったと考えられる。
また、コンパイラオプション（-O3 -march=native）によるSIMD（自動ベクトル化）の最適化が強力に機能するようになった。
スレッド並列化（OpenMP）
最も外側のループの直前に "#pragma omp parallel for" を追加し、計算プロセスを並列化した。
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define drand48() ((double)rand() / (RAND_MAX + 1.0))
#define BLOCK_SIZE 64


void matmul (double* A, double* B, double* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;

            for (int k = 0; k < n; ++k)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

// 自分で実装した行列積コード
void mymatmul (double* A, double* B, double* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double a_ij = A[i * n + j];

            for (int k = 0; k < n; ++k)
                C[i * n + k] += a_ij * B[j * n + k];
        }
    }
} 

double getSec (struct timespec startTime, struct timespec endTime) {
    return (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec) * 1.e-9;
}

double getGFlops (double elapsed_time, int n) {
    double flops = 2.0 * n * n * n;
    double flops_per_second = flops / elapsed_time / 1e9;
    return flops_per_second;
} 

int compare (double *C, double *C_ref, int n) {
    double max_error = 0.0;
    double relative_error;

    for (int i = 0; i < n * n; i++) {
        if (C_ref[i] != 0.0) {
            relative_error = fabs(C[i] - C_ref[i]) / fabs(C_ref[i]);
            if (relative_error > max_error)
                max_error = relative_error;
        }
    }

    if (max_error <= 1e-8) {
        printf("OK (max relative error = %.3e)\n", max_error);
        return 1;
    } else {
        printf("NG (max relative error = %.3e)\n", max_error);
        return 0;
    }
} 

int main(int argc, char *argv[]) {
    int n = atoi (argv[1]);
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));
    double *C_ref = (double *)malloc(n * n * sizeof(double));

    struct timespec startTime, endTime;
    for (int i = 0; i < n * n; i++) A[i] = B[i] = drand48();
    clock_gettime(CLOCK_REALTIME, &startTime);
    
    matmul(A, B, C_ref, n);
    clock_gettime(CLOCK_REALTIME, &endTime);
    double sec_ref = getSec (startTime, endTime);
    clock_gettime(CLOCK_REALTIME, &startTime);

    mymatmul(A, B, C, n);
    clock_gettime(CLOCK_REALTIME, &endTime);
    double sec = getSec (startTime, endTime);

    if (compare (C, C_ref, n)) {
        double gflops = getGFlops (sec, n);
        double speedup = sec_ref/sec;
        printf("n = %d, %.3e [sec], %.3e [GFlops], speedup = %.3e [times]\n", n, sec, gflops, speedup);
    }
    free(A); free(B); free(C); free(C_ref);
}