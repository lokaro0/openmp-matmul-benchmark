#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <errno.h>
#include <limits.h>

#define drand48() ((double)rand() / (RAND_MAX + 1.0))


void matmul_ijk_serial (const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;

            for (int k = 0; k < n; ++k)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
}

void matmul_ikj_serial (const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0;
    }

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double a_ik = A[i * n + k];

            for (int j = 0; j < n; ++j)
                C[i * n + j] += a_ik * B[k * n + j];
        }
    }
}

void matmul_ikj_openmp (const double* A, const double* B, double* C, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double a_ik = A[i * n + k];

            for (int j = 0; j < n; ++j)
                C[i * n + j] += a_ik * B[k * n + j];
        }
    }
}

double calculate_gflops (double elapsed_time, int n) {
    double flops = 2.0 * n * n * n;
    double flops_per_second = flops / elapsed_time / 1e9;
    return flops_per_second;
}

int compare (const double *actual, const double *reference, int n) {
    const double absolute_tolerance = 1e-12;
    const double relative_tolerance = 1e-8;

    size_t element_count = (size_t)n * (size_t)n;

    for (size_t index = 0; index < element_count; index++) {
        double actual_value = actual[index];
        double reference_value = reference[index];

        if (!isfinite(actual_value) || !isfinite(reference_value)) {
            fprintf(
                stderr,
                "NG (non-finite value at index %zu: actual = %.17g, reference = %.17g)\n",
                index, actual_value, reference_value
            );
            return 0;
        }

        double absolute_error =
            fabs(actual_value - reference_value);

        double allowed_error =
            absolute_tolerance
            + relative_tolerance * fabs(reference_value);

        if (absolute_error > allowed_error) {
            fprintf(
                stderr,
                "NG (index = %zu, actual = %.17g, "
                "reference = %.17g, absolute error = %.3e, allowed error = %.3e)\n",
                index, actual_value,
                reference_value, absolute_error, allowed_error
            );
            return 0;
        }
    }

    return 1;
}

typedef void (*matmul_function) (
    const double *A,
    const double *B,
    double *C,
    int n
);

enum {
    MEASUREMENT_COUNT = 5
};

struct benchmark_case {
    const char *name;
    matmul_function function;
    double *output_matrix;
    double elapsed_seconds[MEASUREMENT_COUNT];
};

int main(int argc, char *argv[]) {
    fprintf(stderr, "Timer resolution: %.9e [s]\n", omp_get_wtick());

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    errno = 0;
    char *end_pointer;
    long parsed_size = strtol(argv[1], &end_pointer, 10);

    if (
        errno != 0
        || end_pointer == argv[1]
        || *end_pointer != '\0'
        || parsed_size <= 0
        || parsed_size > INT_MAX
        || parsed_size > INT_MAX / parsed_size
    ) {
        fprintf(stderr,"Error: matrix_size is invalid or too large.\n");
        return EXIT_FAILURE;
    }

    int n = (int)parsed_size;
    int repetitions_per_measurement;

    if (n <= 256) {
        repetitions_per_measurement = 100;
    } else if (n <= 512) {
        repetitions_per_measurement = 20;
    } else {
        repetitions_per_measurement = 5;
    }

    size_t element_count = (size_t)n * (size_t)n;
    size_t matrix_bytes = element_count * sizeof(double);

    double *A = (double *)malloc(matrix_bytes);
    double *B = (double *)malloc(matrix_bytes);
    double *C_ijk_serial = (double *)malloc(matrix_bytes);
    double *C_ikj_serial = (double *)malloc(matrix_bytes);
    double *C_ikj_openmp = (double *)malloc(matrix_bytes);

    if (
        A == NULL
        || B == NULL
        || C_ijk_serial == NULL
        || C_ikj_serial == NULL
        || C_ikj_openmp == NULL
    ) {
        fprintf(stderr, "Error: failed to allocate memory for matrices.\n");

        free(A);
        free(B);
        free(C_ijk_serial);
        free(C_ikj_serial);
        free(C_ikj_openmp);

        return EXIT_FAILURE;
    }

    const unsigned int random_seed = 0;
    srand(random_seed);

    for (size_t index = 0; index < element_count; index++) {
        A[index] = drand48();
        B[index] = drand48();
    }

    /* warm up*/
    matmul_ijk_serial(A, B, C_ijk_serial, n);
    matmul_ikj_serial(A, B, C_ikj_serial, n);
    matmul_ikj_openmp(A, B, C_ikj_openmp, n);

    struct benchmark_case cases[] = {
        {"ijk_serial", matmul_ijk_serial, C_ijk_serial, {0.0}},
        {"ikj_serial", matmul_ikj_serial, C_ikj_serial, {0.0}},
        {"ikj_openmp", matmul_ikj_openmp, C_ikj_openmp, {0.0}},
    };

    int thread_count = omp_get_max_threads();
    size_t case_count = sizeof(cases) / sizeof(cases[0]);

    for (size_t trial_index = 0; trial_index < MEASUREMENT_COUNT; ++trial_index) {
        for (size_t execution_index = 0; execution_index < case_count; ++execution_index) {
            size_t case_index = (trial_index + execution_index) % case_count;
            struct benchmark_case *current_case = &cases[case_index];

            double start_time = omp_get_wtime();

            for (int repetition = 0;
                repetition < repetitions_per_measurement;
                ++repetition) {
                current_case->function(
                    A, B, current_case->output_matrix, n
                );
            }

            double total_seconds = omp_get_wtime() - start_time;

            current_case->elapsed_seconds[trial_index] =
                total_seconds / repetitions_per_measurement;

            double gflops = calculate_gflops(current_case->elapsed_seconds[trial_index], n);
            printf(
                "%zu,%s,%d,%d,%.9e,%.9e\n",
                trial_index + 1, current_case->name, n, thread_count, current_case->elapsed_seconds[trial_index], gflops
            );
        }
    }

    int results_are_correct = 1;

    results_are_correct &= compare(C_ikj_serial, C_ijk_serial, n);
    results_are_correct &= compare(C_ikj_openmp, C_ijk_serial, n);
    if (!results_are_correct) {
        free(A); free(B); free(C_ijk_serial); free(C_ikj_serial); free(C_ikj_openmp);
        return EXIT_FAILURE;
    }

    free(A); free(B); free(C_ijk_serial); free(C_ikj_serial); free(C_ikj_openmp);
    return EXIT_SUCCESS;
}
